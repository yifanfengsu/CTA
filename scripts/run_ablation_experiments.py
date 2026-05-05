#!/usr/bin/env python3
"""Run diagnostic ablation experiments for entry filters."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable


class AblationError(Exception):
    """Raised when ablation experiments cannot continue."""


DEFAULT_FULL_START = "2025-01-01"
DEFAULT_FULL_END = "2026-03-31"
SPLIT_RANGES: dict[str, tuple[str, str]] = {
    "full": (DEFAULT_FULL_START, DEFAULT_FULL_END),
    "train": ("2025-01-01", "2025-09-30"),
    "validation": ("2025-10-01", "2025-12-31"),
    "oos": ("2026-01-01", "2026-03-31"),
}

REPORT_METRICS: tuple[str, ...] = (
    "total_net_pnl",
    "max_ddpercent",
    "sharpe_ratio",
    "engine_trade_count",
    "closed_round_trip_count",
    "win_rate",
    "profit_loss_ratio",
    "bankrupt",
    "statistics_valid",
)


@dataclass(frozen=True, slots=True)
class AblationCandidate:
    """One diagnostic ablation candidate."""

    name: str
    description: str
    setting_overrides: dict[str, Any]
    notes: str = ""


@dataclass(slots=True)
class AblationRun:
    """One candidate and its cost/no-cost report results."""

    candidate: AblationCandidate
    candidate_dir: Path
    no_cost_report_dir: Path
    cost_report_dir: Path
    no_cost_returncode: int | None = None
    cost_returncode: int | None = None
    no_cost_stats: dict[str, Any] | None = None
    no_cost_diagnostics: dict[str, Any] | None = None
    cost_stats: dict[str, Any] | None = None
    cost_diagnostics: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run entry-filter ablation experiments.")
    parser.add_argument("--vt-symbol", default="BTCUSDT_SWAP_OKX.GLOBAL")
    parser.add_argument("--start", help="Backtest start date/datetime. Overrides --split range when set.")
    parser.add_argument("--end", help="Backtest end date/datetime. Overrides --split range when set.")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--base-config", default="config/strategy_sanity_min_size.json")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--rate", type=float, default=0.0005)
    parser.add_argument("--slippage-mode", choices=("ticks", "absolute"), default="ticks")
    parser.add_argument("--slippage", type=float, default=2.0)
    parser.add_argument("--output-dir", help="Default: reports/ablation/YYYYMMDD_HHMMSS")
    parser.add_argument(
        "--data-check-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep strict sqlite history checks on. Default: enabled.",
    )
    parser.add_argument("--max-runs", type=int, default=100, help="Maximum candidate count to evaluate.")
    parser.add_argument("--split", choices=tuple(SPLIT_RANGES), default="full")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_path(path_arg: str | None, default_path: Path | None = None) -> Path:
    """Resolve a path relative to project root."""

    if path_arg:
        path = Path(path_arg)
    elif default_path is not None:
        path = default_path
    else:
        raise AblationError("缺少路径参数且没有默认值")

    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json_file(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    if not path.exists():
        raise AblationError(f"配置文件不存在: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AblationError(f"读取 JSON 失败: {path} | {exc!r}") from exc

    if not isinstance(payload, dict):
        raise AblationError(f"JSON 顶层结构必须是对象: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    """Resolve ablation output directory."""

    if output_dir_arg:
        return resolve_path(output_dir_arg)
    return PROJECT_ROOT / "reports" / "ablation" / datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_requested_range(args: argparse.Namespace) -> tuple[str, str, bool]:
    """Resolve split range, giving explicit start/end priority."""

    split_start, split_end = SPLIT_RANGES[args.split]
    explicit_range = bool(args.start or args.end)
    if explicit_range:
        return args.start or split_start, args.end or split_end, True
    return split_start, split_end, False


def build_default_candidates() -> list[AblationCandidate]:
    """Build the fixed diagnostic ablation shortlist."""

    return [
        AblationCandidate("baseline", "基线 sanity 配置", {}),
        AblationCandidate("long_only", "只允许新开多，禁止新开空", {"enable_short": False}),
        AblationCandidate("short_only", "只允许新开空，禁止新开多", {"enable_long": False}),
        AblationCandidate("no_weekend", "禁止周六/周日新开仓", {"block_weekend_entries": True}),
        AblationCandidate(
            "weekdays_only",
            "只允许周一到周五新开仓",
            {"entry_weekday_allowlist": "0,1,2,3,4"},
        ),
        AblationCandidate(
            "no_worst_hours_from_current_report",
            "屏蔽当前归因报告里的最差小时",
            {"entry_hour_blocklist": "0,4,8,11,12,13,16,18,19,21"},
            "in-sample diagnostic from current full-sample attribution; not a production conclusion",
        ),
        AblationCandidate(
            "no_weekend_no_worst_hours",
            "禁止周末新开仓，并屏蔽当前归因报告里的最差小时",
            {
                "block_weekend_entries": True,
                "entry_hour_blocklist": "0,4,8,11,12,13,16,18,19,21",
            },
            "in-sample diagnostic from current full-sample attribution; not a production conclusion",
        ),
        AblationCandidate(
            "thursday_only",
            "只允许周四新开仓",
            {"entry_weekday_allowlist": "3"},
            "sample-mined / high overfit risk",
        ),
        AblationCandidate(
            "weekday_no_worst_hours",
            "只允许工作日新开仓，并屏蔽当前归因报告里的最差小时",
            {
                "entry_weekday_allowlist": "0,1,2,3,4",
                "entry_hour_blocklist": "0,4,8,11,12,13,16,18,19,21",
            },
            "in-sample diagnostic from current full-sample attribution; not a production conclusion",
        ),
    ]


def with_report_tag(candidate: AblationCandidate) -> dict[str, Any]:
    """Return setting overrides with a report-only tag."""

    overrides = dict(candidate.setting_overrides)
    overrides["entry_filter_tag"] = candidate.name
    return overrides


def build_ablation_runs(output_dir: Path, max_runs: int) -> list[AblationRun]:
    """Build capped ablation run definitions."""

    candidates = build_default_candidates()
    capped_count = max(1, min(int(max_runs), len(candidates)))
    runs: list[AblationRun] = []

    for index, candidate in enumerate(candidates[:capped_count], start=1):
        candidate_dir = output_dir / "candidates" / f"{index:02d}_{candidate.name}"
        runs.append(
            AblationRun(
                candidate=candidate,
                candidate_dir=candidate_dir,
                no_cost_report_dir=candidate_dir / "no_cost",
                cost_report_dir=candidate_dir / "cost",
            )
        )
    return runs


def build_backtest_command(
    args: argparse.Namespace,
    base_config_path: Path,
    start: str,
    end: str,
    output_dir: Path,
    setting_overrides: dict[str, Any],
    rate: float,
    slippage_mode: str,
    slippage: float,
) -> list[str]:
    """Build the backtest command for one cost mode."""

    command = [
        sys.executable,
        "scripts/backtest_okx_mhf.py",
        "--vt-symbol",
        args.vt_symbol,
        "--start",
        start,
        "--end",
        end,
        "--timezone",
        args.timezone,
        "--capital",
        str(args.capital),
        "--rate",
        str(rate),
        "--slippage-mode",
        slippage_mode,
        "--slippage",
        str(slippage),
        "--strategy-config",
        str(base_config_path),
        "--setting-overrides",
        json.dumps(setting_overrides, ensure_ascii=False, sort_keys=True),
        "--output-dir",
        str(output_dir),
    ]

    if args.data_check_strict:
        command.append("--data-check-strict")
    else:
        command.append("--no-data-check-strict")
    return command


def run_backtest_command(
    command: list[str],
    workdir: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    """Execute one backtest command and persist logs."""

    completed = subprocess.run(
        command,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def load_report_json(path: Path) -> dict[str, Any] | None:
    """Load one optional report JSON file."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_candidate_definition(run: AblationRun) -> None:
    """Write one candidate definition file."""

    write_json(
        run.candidate_dir / "candidate_definition.json",
        {
            "name": run.candidate.name,
            "description": run.candidate.description,
            "notes": run.candidate.notes,
            "setting_overrides": with_report_tag(run.candidate),
            "no_cost_report_dir": run.no_cost_report_dir,
            "cost_report_dir": run.cost_report_dir,
        },
    )


def run_one_candidate(
    args: argparse.Namespace,
    base_config_path: Path,
    start: str,
    end: str,
    run: AblationRun,
    logger: logging.Logger,
) -> None:
    """Run no-cost and cost-aware backtests for one candidate."""

    run.candidate_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_definition(run)
    setting_overrides = with_report_tag(run.candidate)

    no_cost_command = build_backtest_command(
        args=args,
        base_config_path=base_config_path,
        start=start,
        end=end,
        output_dir=run.no_cost_report_dir,
        setting_overrides=setting_overrides,
        rate=0.0,
        slippage_mode="absolute",
        slippage=0.0,
    )
    run.no_cost_returncode = run_backtest_command(
        command=no_cost_command,
        workdir=PROJECT_ROOT,
        stdout_path=run.candidate_dir / "no_cost_stdout.log",
        stderr_path=run.candidate_dir / "no_cost_stderr.log",
    )
    run.no_cost_stats = load_report_json(run.no_cost_report_dir / "stats.json")
    run.no_cost_diagnostics = load_report_json(run.no_cost_report_dir / "diagnostics.json")

    cost_command = build_backtest_command(
        args=args,
        base_config_path=base_config_path,
        start=start,
        end=end,
        output_dir=run.cost_report_dir,
        setting_overrides=setting_overrides,
        rate=args.rate,
        slippage_mode=args.slippage_mode,
        slippage=args.slippage,
    )
    run.cost_returncode = run_backtest_command(
        command=cost_command,
        workdir=PROJECT_ROOT,
        stdout_path=run.candidate_dir / "cost_stdout.log",
        stderr_path=run.candidate_dir / "cost_stderr.log",
    )
    run.cost_stats = load_report_json(run.cost_report_dir / "stats.json")
    run.cost_diagnostics = load_report_json(run.cost_report_dir / "diagnostics.json")

    write_json(run.candidate_dir / "candidate_summary.json", summarize_candidate(run))
    log_event(
        logger,
        logging.INFO,
        "ablation.candidate_completed",
        "Completed ablation candidate",
        candidate_name=run.candidate.name,
        no_cost_returncode=run.no_cost_returncode,
        cost_returncode=run.cost_returncode,
        no_cost_total_net_pnl=(run.no_cost_stats or {}).get("total_net_pnl"),
        cost_total_net_pnl=(run.cost_stats or {}).get("total_net_pnl"),
    )


def extract_metric(stats: dict[str, Any] | None, metric: str) -> Any:
    """Extract one report metric from stats."""

    if not stats:
        return None
    return stats.get(metric)


def summarize_candidate(run: AblationRun) -> dict[str, Any]:
    """Build a JSON summary for one candidate."""

    return {
        "name": run.candidate.name,
        "description": run.candidate.description,
        "notes": run.candidate.notes,
        "setting_overrides": with_report_tag(run.candidate),
        "candidate_dir": run.candidate_dir,
        "no_cost_report_dir": run.no_cost_report_dir,
        "no_cost_returncode": run.no_cost_returncode,
        "no_cost_stats": run.no_cost_stats,
        "no_cost_diagnostics": run.no_cost_diagnostics,
        "cost_report_dir": run.cost_report_dir,
        "cost_returncode": run.cost_returncode,
        "cost_stats": run.cost_stats,
        "cost_diagnostics": run.cost_diagnostics,
    }


def build_leaderboard(runs: list[AblationRun]) -> pd.DataFrame:
    """Build the ablation leaderboard table."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "name": run.candidate.name,
            "description": run.candidate.description,
            "notes": run.candidate.notes,
            "setting_overrides": json.dumps(with_report_tag(run.candidate), ensure_ascii=False, sort_keys=True),
            "no_cost_report_dir": str(run.no_cost_report_dir),
            "no_cost_returncode": run.no_cost_returncode,
            "cost_report_dir": str(run.cost_report_dir),
            "cost_returncode": run.cost_returncode,
            "candidate_dir": str(run.candidate_dir),
        }
        for metric in REPORT_METRICS:
            row[f"no_cost_{metric}"] = extract_metric(run.no_cost_stats, metric)
            row[f"cost_{metric}"] = extract_metric(run.cost_stats, metric)
        rows.append(row)

    leaderboard_df = pd.DataFrame(rows)
    if leaderboard_df.empty:
        return leaderboard_df

    leaderboard_df["_sort_no_cost_pnl"] = pd.to_numeric(
        leaderboard_df["no_cost_total_net_pnl"], errors="coerce"
    )
    leaderboard_df["_sort_cost_pnl"] = pd.to_numeric(
        leaderboard_df["cost_total_net_pnl"], errors="coerce"
    )
    leaderboard_df = leaderboard_df.sort_values(
        by=["_sort_no_cost_pnl", "_sort_cost_pnl"],
        ascending=[False, False],
        na_position="last",
    ).drop(columns=["_sort_no_cost_pnl", "_sort_cost_pnl"])
    leaderboard_df = leaderboard_df.reset_index(drop=True)
    leaderboard_df.insert(0, "rank", range(1, len(leaderboard_df.index) + 1))
    return leaderboard_df


def render_metric(value: Any) -> str:
    """Render a metric for Markdown tables."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(
    leaderboard_df: pd.DataFrame,
    split: str,
    start: str,
    end: str,
    explicit_range: bool,
) -> str:
    """Render the Markdown ablation report."""

    lines = [
        "# 策略 Ablation 实验报告",
        "",
        "本报告用于诊断方向、周末、小时和样本切分过滤是否稳定，不等于参数优化结论。",
        "",
        f"- split: {split}",
        f"- start: {start}",
        f"- end: {end}",
        f"- explicit_start_end: {explicit_range}",
        "",
        "关键约束：从 full sample 归因得到的周末/小时过滤只能视为 in-sample diagnostic；必须再看 train、validation、oos 是否一致。",
        "",
    ]

    if leaderboard_df.empty:
        lines.append("没有可用候选结果。")
        return "\n".join(lines) + "\n"

    display_columns = [
        "rank",
        "name",
        "no_cost_total_net_pnl",
        "cost_total_net_pnl",
        "no_cost_max_ddpercent",
        "cost_max_ddpercent",
        "no_cost_sharpe_ratio",
        "cost_sharpe_ratio",
        "no_cost_engine_trade_count",
        "cost_engine_trade_count",
        "notes",
    ]
    lines.extend(
        [
            "## Leaderboard",
            "",
            "| rank | candidate | no-cost pnl | cost pnl | no-cost max dd% | cost max dd% | no-cost sharpe | cost sharpe | no-cost trades | cost trades | notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in leaderboard_df[display_columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    render_metric(row["rank"]),
                    render_metric(row["name"]),
                    render_metric(row["no_cost_total_net_pnl"]),
                    render_metric(row["cost_total_net_pnl"]),
                    render_metric(row["no_cost_max_ddpercent"]),
                    render_metric(row["cost_max_ddpercent"]),
                    render_metric(row["no_cost_sharpe_ratio"]),
                    render_metric(row["cost_sharpe_ratio"]),
                    render_metric(row["no_cost_engine_trade_count"]),
                    render_metric(row["cost_engine_trade_count"]),
                    render_metric(row["notes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- no-cost 仍为负：没有毛 alpha，不能进入 OKX DEMO。",
            "- no-cost 为正但 cost 为负：成本拖累或交易频率仍不可接受。",
            "- full 为正但 oos 为负：高度疑似过拟合。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_ablation(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    """Run the full ablation workflow."""

    base_config_path = resolve_path(args.base_config)
    base_config = load_json_file(base_config_path)
    if not isinstance(base_config.get("setting"), dict):
        raise AblationError(f"base-config 缺少 setting 对象: {base_config_path}")

    start, end, explicit_range = resolve_requested_range(args)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = build_ablation_runs(output_dir=output_dir, max_runs=args.max_runs)
    if not runs:
        raise AblationError("没有生成任何 ablation 候选")

    write_json(
        output_dir / "ablation_config.json",
        {
            "base_config_path": base_config_path,
            "vt_symbol": args.vt_symbol,
            "split": args.split,
            "start": start,
            "end": end,
            "explicit_start_end": explicit_range,
            "timezone": args.timezone,
            "capital": args.capital,
            "rate": args.rate,
            "slippage_mode": args.slippage_mode,
            "slippage": args.slippage,
            "data_check_strict": args.data_check_strict,
            "max_runs": args.max_runs,
            "candidate_names": [run.candidate.name for run in runs],
        },
    )

    for run in runs:
        run_one_candidate(
            args=args,
            base_config_path=base_config_path,
            start=start,
            end=end,
            run=run,
            logger=logger,
        )

    leaderboard_df = build_leaderboard(runs)
    leaderboard_path = output_dir / "ablation_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False, encoding="utf-8")
    report_path = output_dir / "ablation_report.md"
    report_path.write_text(
        render_report(
            leaderboard_df=leaderboard_df,
            split=args.split,
            start=start,
            end=end,
            explicit_range=explicit_range,
        ),
        encoding="utf-8",
    )

    summary = {
        "output_dir": output_dir,
        "base_config_path": base_config_path,
        "vt_symbol": args.vt_symbol,
        "split": args.split,
        "start": start,
        "end": end,
        "explicit_start_end": explicit_range,
        "candidate_count": len(runs),
        "backtest_run_count": len(runs) * 2,
        "leaderboard_csv": leaderboard_path,
        "ablation_report_md": report_path,
        "candidates": [summarize_candidate(run) for run in runs],
        "top_10": leaderboard_df.head(min(10, len(leaderboard_df.index))).to_dict(orient="records")
        if not leaderboard_df.empty
        else [],
    }
    write_json(output_dir / "ablation_summary.json", summary)

    log_event(
        logger,
        logging.INFO,
        "ablation.completed",
        "Ablation experiments completed",
        output_dir=output_dir,
        split=args.split,
        start=start,
        end=end,
        candidate_count=len(runs),
    )
    return summary


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("run_ablation_experiments", verbose=args.verbose)

    try:
        summary = run_ablation(args=args, logger=logger)
        print_json_block("Ablation summary:", summary)
        return 0
    except AblationError as exc:
        log_event(logger, logging.ERROR, "ablation.error", str(exc))
        return 1
    except Exception:
        logger.exception(
            "Unexpected error during ablation experiments",
            extra={"event": "ablation.unexpected_error"},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
