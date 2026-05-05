#!/usr/bin/env python3
"""Compare HTF signal research across train/validation/OOS splits."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "htf_compare"
PRIMARY_RETURN_COLUMN = "median_future_return_120m"
PRIMARY_EXPECTANCY_COLUMN = "best_expectancy_r"


class HtfSignalCompareError(Exception):
    """Raised when HTF comparison cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Compare HTF signal research across sample splits.")
    parser.add_argument("--train-dir", required=True, help="Train htf_signals directory.")
    parser.add_argument("--validation-dir", required=True, help="Validation htf_signals directory.")
    parser.add_argument("--oos-dir", required=True, help="OOS htf_signals directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: reports/research/htf_compare.",
    )
    parser.add_argument("--json", action="store_true", help="Print htf_compare_summary.json payload to stdout.")
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


def sign_label(value: Any) -> str:
    """Return positive, negative, zero, or unknown."""

    number = finite_or_none(value)
    if number is None:
        return "unknown"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def read_split_leaderboard(split: str, directory: Path) -> pd.DataFrame:
    """Read one split htf_policy_leaderboard.csv."""

    path = directory / "htf_policy_leaderboard.csv"
    if not path.exists():
        raise HtfSignalCompareError(f"{split} 缺少 htf_policy_leaderboard.csv: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise HtfSignalCompareError(f"读取 {split} leaderboard 失败: {exc!r}") from exc
    if "policy_name" not in frame.columns:
        raise HtfSignalCompareError(f"{split} leaderboard 缺少 policy_name 列")
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


def build_compare_leaderboard(all_df: pd.DataFrame) -> pd.DataFrame:
    """Build one cross-split row per policy."""

    rows: list[dict[str, Any]] = []
    for policy_name, group_df in all_df.groupby("policy_name", dropna=False):
        row: dict[str, Any] = {"policy_name": policy_name}
        return_signs: list[str] = []
        expectancy_signs: list[str] = []
        effective_splits: list[str] = []
        available_return_count = 0
        available_expectancy_count = 0
        for split in SPLITS:
            split_data = split_row(group_df, split)
            signal_count = finite_or_none(split_data.get("signal_count")) if split_data is not None else None
            median_return = finite_or_none(split_data.get(PRIMARY_RETURN_COLUMN)) if split_data is not None else None
            expectancy = finite_or_none(split_data.get(PRIMARY_EXPECTANCY_COLUMN)) if split_data is not None else None
            positive_rate = finite_or_none(split_data.get("positive_rate_120m")) if split_data is not None else None
            row[f"{split}_signal_count"] = int(signal_count) if signal_count is not None else None
            row[f"{split}_{PRIMARY_RETURN_COLUMN}"] = median_return
            row[f"{split}_{PRIMARY_EXPECTANCY_COLUMN}"] = expectancy
            row[f"{split}_positive_rate_120m"] = positive_rate
            return_sign = sign_label(median_return)
            expectancy_sign = sign_label(expectancy)
            row[f"{split}_return_direction"] = return_sign
            row[f"{split}_expectancy_direction"] = expectancy_sign
            if return_sign not in {"unknown", "zero"}:
                return_signs.append(return_sign)
                available_return_count += 1
            if expectancy_sign not in {"unknown", "zero"}:
                expectancy_signs.append(expectancy_sign)
                available_expectancy_count += 1
            has_signals = bool(signal_count is not None and signal_count > 0)
            if has_signals and median_return is not None and median_return > 0 and expectancy is not None and expectancy > 0:
                effective_splits.append(split)

        row["available_return_split_count"] = int(available_return_count)
        row["available_expectancy_split_count"] = int(available_expectancy_count)
        row["return_direction_consistent"] = bool(available_return_count == len(SPLITS) and len(set(return_signs)) == 1)
        row["expectancy_direction_consistent"] = bool(
            available_expectancy_count == len(SPLITS) and len(set(expectancy_signs)) == 1
        )
        row["consistent_return_direction"] = return_signs[0] if row["return_direction_consistent"] else None
        row["consistent_expectancy_direction"] = expectancy_signs[0] if row["expectancy_direction_consistent"] else None
        row["positive_split_count"] = int(len(effective_splits))
        row["positive_splits"] = ",".join(effective_splits)
        row["stable_candidate"] = bool(len(effective_splits) == len(SPLITS))
        row["single_split_only"] = bool(len(effective_splits) == 1)
        row["strategy_v2_candidate"] = bool(row["stable_candidate"])
        rows.append(row)

    columns = [
        "policy_name",
        "train_signal_count",
        "validation_signal_count",
        "oos_signal_count",
        f"train_{PRIMARY_RETURN_COLUMN}",
        f"validation_{PRIMARY_RETURN_COLUMN}",
        f"oos_{PRIMARY_RETURN_COLUMN}",
        f"train_{PRIMARY_EXPECTANCY_COLUMN}",
        f"validation_{PRIMARY_EXPECTANCY_COLUMN}",
        f"oos_{PRIMARY_EXPECTANCY_COLUMN}",
        "train_positive_rate_120m",
        "validation_positive_rate_120m",
        "oos_positive_rate_120m",
        "train_return_direction",
        "validation_return_direction",
        "oos_return_direction",
        "train_expectancy_direction",
        "validation_expectancy_direction",
        "oos_expectancy_direction",
        "available_return_split_count",
        "available_expectancy_split_count",
        "return_direction_consistent",
        "expectancy_direction_consistent",
        "consistent_return_direction",
        "consistent_expectancy_direction",
        "positive_split_count",
        "positive_splits",
        "stable_candidate",
        "single_split_only",
        "strategy_v2_candidate",
    ]
    return pd.DataFrame(rows, columns=columns)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def build_summary(
    split_dirs: dict[str, Path],
    output_dir: Path,
    compare_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build htf_compare_summary.json payload."""

    direction_consistent = compare_df[compare_df["return_direction_consistent"] == True].copy()  # noqa: E712
    single_split = compare_df[compare_df["single_split_only"] == True].copy()  # noqa: E712
    stable = compare_df[compare_df["stable_candidate"] == True].copy()  # noqa: E712
    no_stable = bool(stable.empty)
    return {
        "split_dirs": {split: str(path) for split, path in split_dirs.items()},
        "output_dir": str(output_dir),
        "primary_return_column": PRIMARY_RETURN_COLUMN,
        "primary_expectancy_column": PRIMARY_EXPECTANCY_COLUMN,
        "policy_count": int(compare_df["policy_name"].nunique()) if not compare_df.empty else 0,
        "direction_consistent_policies": dataframe_records(
            direction_consistent.sort_values(["stable_candidate", "policy_name"], ascending=[False, True])
        ),
        "single_split_only_policies": dataframe_records(
            single_split.sort_values(["positive_split_count", "policy_name"], ascending=[False, True])
        ),
        "stable_strategy_v2_candidates": dataframe_records(
            stable.sort_values(["policy_name"], ascending=[True])
        ),
        "no_stable_htf_policy": no_stable,
        "strategy_v2_candidate_exists": bool(not no_stable),
        "overfit_risk": bool(not single_split.empty),
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format numeric values."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def format_policy_rows(rows: list[dict[str, Any]], limit: int = 12) -> str:
    """Format comparison rows."""

    if not rows:
        return "- 无"
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"- {row.get('policy_name')}: "
            f"return_direction={row.get('consistent_return_direction')}, "
            f"train_120m={format_number(row.get('train_median_future_return_120m'))}, "
            f"validation_120m={format_number(row.get('validation_median_future_return_120m'))}, "
            f"oos_120m={format_number(row.get('oos_median_future_return_120m'))}, "
            f"positive_splits={row.get('positive_splits')}"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render htf_compare_report.md."""

    direction_consistent = summary.get("direction_consistent_policies") or []
    single_split = summary.get("single_split_only_policies") or []
    stable = summary.get("stable_strategy_v2_candidates") or []
    no_stable = bool(summary.get("no_stable_htf_policy"))
    candidate_answer = "没有稳定候选进入 Strategy V2" if no_stable else "存在稳定候选可进入 Strategy V2 设计评审"
    return (
        "# HTF Signal Research 跨样本比较\n\n"
        "## 核心结论\n"
        f"- no_stable_htf_policy={str(no_stable).lower()}\n"
        f"- overfit_risk={str(bool(summary.get('overfit_risk'))).lower()}\n"
        f"- {candidate_answer}\n\n"
        "## 哪些 policy 在三段方向一致？\n"
        f"{format_policy_rows(direction_consistent)}\n\n"
        "## 哪些 policy 只在某一段有效？\n"
        f"{format_policy_rows(single_split)}\n\n"
        "## 是否有稳定候选进入 Strategy V2？\n"
        f"{format_policy_rows(stable)}\n\n"
        "## 输出文件\n"
        "- htf_compare_summary.json\n"
        "- htf_compare_leaderboard.csv\n"
        "- htf_compare_report.md\n"
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
    markdown = render_markdown(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "htf_compare_summary.json", summary)
    write_dataframe(output_dir / "htf_compare_leaderboard.csv", compare_df)
    (output_dir / "htf_compare_report.md").write_text(markdown, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("compare_htf_signal_research", verbose=args.verbose)
    try:
        train_dir = resolve_path(args.train_dir)
        validation_dir = resolve_path(args.validation_dir)
        oos_dir = resolve_path(args.oos_dir)
        output_dir = resolve_path(args.output_dir)
        summary = run_compare(train_dir, validation_dir, oos_dir, output_dir)
        print_json_block(
            "HTF signal comparison summary:",
            {
                "output_dir": output_dir,
                "no_stable_htf_policy": summary.get("no_stable_htf_policy"),
                "stable_strategy_v2_candidates": summary.get("stable_strategy_v2_candidates"),
                "overfit_risk": summary.get("overfit_risk"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except HtfSignalCompareError as exc:
        log_event(logger, logging.ERROR, "htf_signal_compare.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during HTF signal comparison",
            extra={"event": "htf_signal_compare.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
