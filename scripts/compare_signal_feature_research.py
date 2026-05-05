#!/usr/bin/env python3
"""Compare Signal Lab feature research across train/validation/OOS splits."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "feature_compare"
STABLE_ABS_IC_THRESHOLD = 0.05
OVERFIT_ABS_IC_THRESHOLD = 0.10


class SignalFeatureCompareError(Exception):
    """Raised when feature comparison cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Compare Signal Lab feature research across sample splits.")
    parser.add_argument("--train-dir", required=True, help="Train signal_feature_research directory.")
    parser.add_argument("--validation-dir", required=True, help="Validation signal_feature_research directory.")
    parser.add_argument("--oos-dir", required=True, help="OOS signal_feature_research directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: reports/research/feature_compare.",
    )
    parser.add_argument("--json", action="store_true", help="Print feature_compare_summary.json payload to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose structured logs.")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve paths relative to project root."""

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
    """Return positive/negative/zero/unknown sign label."""

    number = finite_or_none(value)
    if number is None:
        return "unknown"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def read_split_frame(split: str, directory: Path, filename: str, required: bool) -> pd.DataFrame:
    """Read one split artifact."""

    path = directory / filename
    if not path.exists():
        if required:
            raise SignalFeatureCompareError(f"{split} 缺少 {filename}: {path}")
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SignalFeatureCompareError(f"读取 {split}/{filename} 失败: {exc!r}") from exc
    frame["split"] = split
    frame["source_dir"] = str(directory)
    return frame


def read_split_outputs(split_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read IC and bin outputs from all splits."""

    ic_frames = []
    bin_frames = []
    for split in SPLITS:
        directory = split_dirs[split]
        ic_frames.append(read_split_frame(split, directory, "feature_ic.csv", required=True))
        bin_frames.append(read_split_frame(split, directory, "feature_bins.csv", required=False))
    ic_df = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    bins_df = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    return ic_df, bins_df


def build_compare_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Build split-comparison rows from feature_ic.csv files."""

    required = {"feature", "target", "split", "spearman", "count"}
    missing = sorted(required - set(ic_df.columns))
    if missing:
        raise SignalFeatureCompareError(f"feature_ic.csv 缺少列: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for (feature, target), group_df in ic_df.groupby(["feature", "target"], dropna=False):
        row: dict[str, Any] = {"feature": feature, "target": target}
        signs: list[str] = []
        abs_values: list[float] = []
        available_count = 0
        for split in SPLITS:
            split_row = group_df[group_df["split"] == split]
            spearman = finite_or_none(split_row.iloc[0].get("spearman")) if not split_row.empty else None
            count = finite_or_none(split_row.iloc[0].get("count")) if not split_row.empty else None
            row[f"{split}_spearman"] = spearman
            row[f"{split}_count"] = int(count) if count is not None else None
            sign = sign_label(spearman)
            row[f"{split}_direction"] = sign
            if sign not in {"unknown", "zero"}:
                signs.append(sign)
                abs_values.append(abs(float(spearman)))
                available_count += 1

        direction_consistent = bool(available_count == len(SPLITS) and len(set(signs)) == 1)
        min_abs = min(abs_values) if len(abs_values) == len(SPLITS) else None
        max_abs = max(abs_values) if abs_values else None
        row["available_split_count"] = int(available_count)
        row["direction_consistent"] = direction_consistent
        row["consistent_direction"] = signs[0] if direction_consistent else None
        row["min_abs_spearman"] = min_abs
        row["max_abs_spearman"] = max_abs
        row["stable_candidate"] = bool(
            target == "future_return_60m"
            and direction_consistent
            and min_abs is not None
            and min_abs >= STABLE_ABS_IC_THRESHOLD
        )
        row["single_split_only"] = bool(
            target == "future_return_60m"
            and max_abs is not None
            and max_abs >= OVERFIT_ABS_IC_THRESHOLD
            and not row["stable_candidate"]
        )
        rows.append(row)

    columns = [
        "feature",
        "target",
        "train_spearman",
        "validation_spearman",
        "oos_spearman",
        "train_count",
        "validation_count",
        "oos_count",
        "train_direction",
        "validation_direction",
        "oos_direction",
        "available_split_count",
        "direction_consistent",
        "consistent_direction",
        "min_abs_spearman",
        "max_abs_spearman",
        "stable_candidate",
        "single_split_only",
    ]
    return pd.DataFrame(rows, columns=columns)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.replace({np.nan: None})
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def build_summary(
    split_dirs: dict[str, Path],
    output_dir: Path,
    compare_ic_df: pd.DataFrame,
    bins_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build feature_compare_summary.json payload."""

    future_df = compare_ic_df[compare_ic_df["target"] == "future_return_60m"].copy()
    stable_df = future_df[future_df["stable_candidate"] == True].copy()  # noqa: E712
    consistent_df = future_df[future_df["direction_consistent"] == True].copy()  # noqa: E712
    single_split_df = future_df[future_df["single_split_only"] == True].copy()  # noqa: E712
    no_stable_feature_edge = bool(stable_df.empty)
    return {
        "split_dirs": {split: str(path) for split, path in split_dirs.items()},
        "output_dir": str(output_dir),
        "stable_abs_ic_threshold": STABLE_ABS_IC_THRESHOLD,
        "overfit_abs_ic_threshold": OVERFIT_ABS_IC_THRESHOLD,
        "feature_count": int(future_df["feature"].nunique()) if not future_df.empty else 0,
        "direction_consistent_features": dataframe_records(
            consistent_df.sort_values(["min_abs_spearman", "feature"], ascending=[False, True])
        ),
        "single_split_only_features": dataframe_records(
            single_split_df.sort_values(["max_abs_spearman", "feature"], ascending=[False, True])
        ),
        "stable_feature_candidates": dataframe_records(
            stable_df.sort_values(["min_abs_spearman", "feature"], ascending=[False, True])
        ),
        "no_stable_feature_edge": no_stable_feature_edge,
        "feature_bins_rows_loaded": int(len(bins_df.index)),
    }


def format_number(value: Any, digits: int = 4) -> str:
    """Format numeric values for reports."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def format_feature_rows(rows: list[dict[str, Any]], limit: int = 8) -> str:
    """Format feature comparison rows for Markdown."""

    if not rows:
        return "- 无"
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"- {row.get('feature')}: direction={row.get('consistent_direction')}, "
            f"train={format_number(row.get('train_spearman'))}, "
            f"validation={format_number(row.get('validation_spearman'))}, "
            f"oos={format_number(row.get('oos_spearman'))}, "
            f"min_abs_ic={format_number(row.get('min_abs_spearman'))}"
        )
    return "\n".join(lines)


def format_single_split_rows(rows: list[dict[str, Any]], limit: int = 8) -> str:
    """Format likely overfit rows."""

    if not rows:
        return "- 无"
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"- {row.get('feature')}: train={format_number(row.get('train_spearman'))}, "
            f"validation={format_number(row.get('validation_spearman'))}, "
            f"oos={format_number(row.get('oos_spearman'))}, "
            f"max_abs_ic={format_number(row.get('max_abs_spearman'))}"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a human-readable comparison report."""

    consistent = summary.get("direction_consistent_features") or []
    single_split = summary.get("single_split_only_features") or []
    stable = summary.get("stable_feature_candidates") or []
    no_edge = bool(summary.get("no_stable_feature_edge"))
    if no_edge:
        candidate_answer = "没有发现可进入策略候选的稳定特征"
    else:
        candidate_answer = "存在跨 train/validation/oos 方向一致且达到 IC 阈值的候选特征"

    return (
        "# Signal Lab 跨样本稳定性比较\n\n"
        "## 核心结论\n"
        f"- no_stable_feature_edge={str(no_edge).lower()}\n"
        f"- {candidate_answer}\n\n"
        "## 哪些特征在 train / validation / oos 方向一致？\n"
        f"{format_feature_rows(consistent)}\n\n"
        "## 哪些特征只在某一段有效，疑似过拟合？\n"
        f"{format_single_split_rows(single_split)}\n\n"
        "## 是否存在可进入策略候选的稳定特征？\n"
        f"{format_feature_rows(stable)}\n\n"
        "## 输出文件\n"
        "- feature_compare_summary.json\n"
        "- feature_compare_ic.csv\n"
        "- feature_compare_report.md\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write CSV with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def run_compare(
    train_dir: Path,
    validation_dir: Path,
    oos_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the cross-sample feature comparison workflow."""

    split_dirs = {"train": train_dir, "validation": validation_dir, "oos": oos_dir}
    ic_df, bins_df = read_split_outputs(split_dirs)
    compare_ic_df = build_compare_ic(ic_df)
    summary = build_summary(split_dirs, output_dir, compare_ic_df, bins_df)
    markdown = render_markdown(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "feature_compare_summary.json", summary)
    write_dataframe(output_dir / "feature_compare_ic.csv", compare_ic_df)
    (output_dir / "feature_compare_report.md").write_text(markdown, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("compare_signal_feature_research", verbose=args.verbose)

    try:
        train_dir = resolve_path(args.train_dir)
        validation_dir = resolve_path(args.validation_dir)
        oos_dir = resolve_path(args.oos_dir)
        output_dir = resolve_path(args.output_dir)
        summary = run_compare(
            train_dir=train_dir,
            validation_dir=validation_dir,
            oos_dir=oos_dir,
            output_dir=output_dir,
        )
        print_json_block(
            "Signal feature comparison summary:",
            {
                "output_dir": output_dir,
                "no_stable_feature_edge": summary.get("no_stable_feature_edge"),
                "stable_feature_candidates": summary.get("stable_feature_candidates"),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except SignalFeatureCompareError as exc:
        log_event(logger, logging.ERROR, "signal_feature_compare.error", str(exc))
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during signal feature comparison",
            extra={"event": "signal_feature_compare.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
