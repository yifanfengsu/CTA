#!/usr/bin/env python3
"""Run a guarded alpha sweep focused on frequency reduction and signal quality."""

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


class AlphaSweepError(Exception):
    """Raised when the alpha sweep cannot continue."""


NO_COST_TOP_COUNT: int = 5
SHORTLIST_CANDIDATES: list[tuple[str, str, dict[str, Any]]] = [
    ("baseline", "基线 sanity 配置", {}),
    ("higher_breakout", "更高 breakout_window", {"breakout_window": 60}),
    ("longer_cooldown", "更长 cooldown_bars", {"cooldown_bars": 20}),
    ("strict_rsi", "更严格 RSI", {"rsi_long": 60, "rsi_short": 40}),
    ("higher_vol_floor", "更高 vol_floor", {"vol_floor": 0.0015}),
    ("lower_vol_ceiling", "更低 vol_ceiling", {"vol_ceiling": 0.01}),
    ("shorter_max_hold", "更短 max_hold_bars", {"max_hold_bars": 10}),
    ("higher_take_profit", "更高 take_profit_atr", {"take_profit_atr": 3.0}),
    ("wider_stop_trail", "更宽 stop/trail", {"stop_atr": 1.5, "trail_atr": 2.2}),
    (
        "conservative_combo",
        "组合型 conservative",
        {
            "breakout_window": 90,
            "cooldown_bars": 20,
            "max_hold_bars": 10,
            "rsi_long": 60,
            "rsi_short": 40,
            "vol_floor": 0.0015,
            "vol_ceiling": 0.01,
            "take_profit_atr": 3.0,
            "stop_atr": 1.5,
            "trail_atr": 2.2,
        },
    ),
]


@dataclass(slots=True)
class CandidateRun:
    """One candidate plus its backtest results."""

    name: str
    description: str
    setting_overrides: dict[str, Any]
    candidate_dir: Path
    cost_report_dir: Path
    cost_stats: dict[str, Any] | None = None
    cost_diagnostics: dict[str, Any] | None = None
    cost_returncode: int | None = None
    cost_score: float | None = None
    no_cost_report_dir: Path | None = None
    no_cost_stats: dict[str, Any] | None = None
    no_cost_diagnostics: dict[str, Any] | None = None
    no_cost_returncode: int | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run a lightweight guarded alpha sweep.")
    parser.add_argument("--vt-symbol", default="BTCUSDT_SWAP_OKX.GLOBAL")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--base-config", default="config/strategy_sanity_min_size.json")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--rate", type=float, default=0.0005)
    parser.add_argument("--slippage-mode", choices=("ticks", "absolute"), default="ticks")
    parser.add_argument("--slippage", type=float, default=2.0)
    parser.add_argument("--output-dir", help="Default: reports/alpha_sweep/YYYYMMDD_HHMMSS")
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument(
        "--no-cost-also",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run extra no-cost checks for top cost candidates. Default: enabled.",
    )
    parser.add_argument(
        "--data-check-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always keep strict history checks on. Default: enabled.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_path(path_arg: str | None, default_path: Path | None = None) -> Path:
    """Resolve a path relative to project root."""

    if path_arg:
        path = Path(path_arg)
    elif default_path is not None:
        path = default_path
    else:
        raise AlphaSweepError("缺少路径参数且没有默认值")

    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json_file(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    if not path.exists():
        raise AlphaSweepError(f"配置文件不存在: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AlphaSweepError(f"读取 JSON 失败: {path} | {exc!r}") from exc

    if not isinstance(payload, dict):
        raise AlphaSweepError(f"JSON 顶层结构必须是对象: {path}")
    return payload


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    """Resolve sweep output directory."""

    if output_dir_arg:
        return resolve_path(output_dir_arg)
    return PROJECT_ROOT / "reports" / "alpha_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S")


def enforce_guardrails(setting: dict[str, Any]) -> dict[str, Any]:
    """Enforce hard risk limits for sweep candidates."""

    guarded = dict(setting)
    guarded["fixed_size"] = 0.01
    guarded["risk_per_trade"] = min(float(guarded.get("risk_per_trade", 0.0005) or 0.0005), 0.0005)
    guarded["max_leverage"] = min(float(guarded.get("max_leverage", 0.5) or 0.5), 0.5)
    guarded["max_notional_ratio"] = min(float(guarded.get("max_notional_ratio", 0.5) or 0.5), 0.5)
    guarded["max_trades_per_day"] = min(int(guarded.get("max_trades_per_day", 10) or 10), 10)
    return guarded


def build_candidate_runs(
    base_setting: dict[str, Any],
    output_dir: Path,
    max_runs: int,
) -> list[CandidateRun]:
    """Build a bounded shortlist of sweep candidates."""

    candidate_runs: list[CandidateRun] = []
    capped_max_runs = max(1, min(max_runs, len(SHORTLIST_CANDIDATES)))

    for index, (name, description, overrides) in enumerate(SHORTLIST_CANDIDATES[:capped_max_runs], start=1):
        candidate_setting = enforce_guardrails({**base_setting, **overrides})
        candidate_dir = output_dir / "candidates" / f"{index:02d}_{name}"
        candidate_runs.append(
            CandidateRun(
                name=name,
                description=description,
                setting_overrides=candidate_setting,
                candidate_dir=candidate_dir,
                cost_report_dir=candidate_dir / "cost",
            )
        )

    return candidate_runs


def build_backtest_command(
    args: argparse.Namespace,
    base_config_path: Path,
    output_dir: Path,
    setting_overrides: dict[str, Any],
    rate: float,
    slippage: float,
) -> list[str]:
    """Build the backtest command for one candidate."""

    command = [
        sys.executable,
        "scripts/backtest_okx_mhf.py",
        "--vt-symbol",
        args.vt_symbol,
        "--start",
        args.start,
        "--end",
        args.end,
        "--timezone",
        args.timezone,
        "--capital",
        str(args.capital),
        "--rate",
        str(rate),
        "--slippage-mode",
        args.slippage_mode,
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


def compute_cost_score(stats: dict[str, Any] | None) -> float:
    """Score a candidate using cost-version stats only."""

    if not stats:
        return -1e12
    if bool(stats.get("bankrupt", True)) or not bool(stats.get("statistics_valid", False)):
        return -1e12

    total_net_pnl = float(stats.get("total_net_pnl", 0.0) or 0.0)
    sharpe_ratio = float(stats.get("sharpe_ratio", 0.0) or 0.0)
    max_ddpercent = abs(float(stats.get("max_ddpercent", 0.0) or 0.0))
    engine_trade_count = float(stats.get("engine_trade_count", 0.0) or 0.0)
    closed_round_trip_count = float(stats.get("closed_round_trip_count", 0.0) or 0.0)

    frequency_penalty = max(engine_trade_count - 6000.0, 0.0) * 0.002
    round_trip_penalty = max(closed_round_trip_count - 3000.0, 0.0) * 0.002

    return (
        total_net_pnl * 10.0
        + sharpe_ratio * 5.0
        - max_ddpercent * 20.0
        - frequency_penalty
        - round_trip_penalty
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_candidate_definition(candidate: CandidateRun) -> None:
    """Write one candidate definition file."""

    write_json(
        candidate.candidate_dir / "candidate_definition.json",
        {
            "name": candidate.name,
            "description": candidate.description,
            "setting_overrides": candidate.setting_overrides,
        },
    )


def summarize_candidate(candidate: CandidateRun) -> dict[str, Any]:
    """Build one candidate summary payload."""

    return {
        "name": candidate.name,
        "description": candidate.description,
        "setting_overrides": candidate.setting_overrides,
        "cost_report_dir": candidate.cost_report_dir,
        "cost_returncode": candidate.cost_returncode,
        "cost_score": candidate.cost_score,
        "cost_stats": candidate.cost_stats,
        "cost_diagnostics": candidate.cost_diagnostics,
        "no_cost_report_dir": candidate.no_cost_report_dir,
        "no_cost_returncode": candidate.no_cost_returncode,
        "no_cost_stats": candidate.no_cost_stats,
        "no_cost_diagnostics": candidate.no_cost_diagnostics,
    }


def run_cost_backtests(
    args: argparse.Namespace,
    base_config_path: Path,
    candidates: list[CandidateRun],
    logger: logging.Logger,
) -> None:
    """Run cost-version backtests for all shortlist candidates."""

    for candidate in candidates:
        candidate.candidate_dir.mkdir(parents=True, exist_ok=True)
        write_candidate_definition(candidate)

        command = build_backtest_command(
            args=args,
            base_config_path=base_config_path,
            output_dir=candidate.cost_report_dir,
            setting_overrides=candidate.setting_overrides,
            rate=args.rate,
            slippage=args.slippage,
        )
        candidate.cost_returncode = run_backtest_command(
            command=command,
            workdir=PROJECT_ROOT,
            stdout_path=candidate.candidate_dir / "cost_stdout.log",
            stderr_path=candidate.candidate_dir / "cost_stderr.log",
        )
        candidate.cost_stats = load_report_json(candidate.cost_report_dir / "stats.json")
        candidate.cost_diagnostics = load_report_json(candidate.cost_report_dir / "diagnostics.json")
        candidate.cost_score = compute_cost_score(candidate.cost_stats)

        write_json(candidate.candidate_dir / "candidate_summary.json", summarize_candidate(candidate))
        log_event(
            logger,
            logging.INFO,
            "alpha_sweep.cost_candidate_completed",
            "Completed cost-version candidate backtest",
            candidate_name=candidate.name,
            returncode=candidate.cost_returncode,
            score=candidate.cost_score,
            total_net_pnl=(candidate.cost_stats or {}).get("total_net_pnl"),
        )


def select_top_cost_candidates(candidates: list[CandidateRun], top_count: int) -> list[CandidateRun]:
    """Select top valid cost candidates for no-cost follow-up."""

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.cost_stats
        and not bool(candidate.cost_stats.get("bankrupt", True))
        and bool(candidate.cost_stats.get("statistics_valid", False))
    ]
    return sorted(
        valid_candidates,
        key=lambda candidate: candidate.cost_score if candidate.cost_score is not None else -1e12,
        reverse=True,
    )[:top_count]


def run_no_cost_backtests(
    args: argparse.Namespace,
    base_config_path: Path,
    candidates: list[CandidateRun],
    logger: logging.Logger,
) -> None:
    """Run no-cost checks for top candidates only."""

    for candidate in candidates:
        candidate.no_cost_report_dir = candidate.candidate_dir / "no_cost"
        command = build_backtest_command(
            args=args,
            base_config_path=base_config_path,
            output_dir=candidate.no_cost_report_dir,
            setting_overrides=candidate.setting_overrides,
            rate=0.0,
            slippage=0.0,
        )
        candidate.no_cost_returncode = run_backtest_command(
            command=command,
            workdir=PROJECT_ROOT,
            stdout_path=candidate.candidate_dir / "no_cost_stdout.log",
            stderr_path=candidate.candidate_dir / "no_cost_stderr.log",
        )
        candidate.no_cost_stats = load_report_json(candidate.no_cost_report_dir / "stats.json")
        candidate.no_cost_diagnostics = load_report_json(candidate.no_cost_report_dir / "diagnostics.json")
        write_json(candidate.candidate_dir / "candidate_summary.json", summarize_candidate(candidate))
        log_event(
            logger,
            logging.INFO,
            "alpha_sweep.no_cost_candidate_completed",
            "Completed no-cost follow-up candidate backtest",
            candidate_name=candidate.name,
            returncode=candidate.no_cost_returncode,
            total_net_pnl=(candidate.no_cost_stats or {}).get("total_net_pnl"),
        )


def build_leaderboard(candidates: list[CandidateRun]) -> pd.DataFrame:
    """Build the sweep leaderboard."""

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        cost_stats = candidate.cost_stats or {}
        no_cost_stats = candidate.no_cost_stats or {}
        rows.append(
            {
                "name": candidate.name,
                "description": candidate.description,
                "cost_score": candidate.cost_score,
                "cost_returncode": candidate.cost_returncode,
                "cost_bankrupt": cost_stats.get("bankrupt"),
                "cost_statistics_valid": cost_stats.get("statistics_valid"),
                "cost_total_net_pnl": cost_stats.get("total_net_pnl"),
                "cost_sharpe_ratio": cost_stats.get("sharpe_ratio"),
                "cost_max_ddpercent": cost_stats.get("max_ddpercent"),
                "cost_engine_trade_count": cost_stats.get("engine_trade_count"),
                "cost_closed_round_trip_count": cost_stats.get("closed_round_trip_count"),
                "no_cost_returncode": candidate.no_cost_returncode,
                "no_cost_total_net_pnl": no_cost_stats.get("total_net_pnl"),
                "no_cost_sharpe_ratio": no_cost_stats.get("sharpe_ratio"),
                "no_cost_bankrupt": no_cost_stats.get("bankrupt"),
                "no_cost_statistics_valid": no_cost_stats.get("statistics_valid"),
                "candidate_dir": str(candidate.candidate_dir),
                "cost_report_dir": str(candidate.cost_report_dir),
                "no_cost_report_dir": str(candidate.no_cost_report_dir) if candidate.no_cost_report_dir else None,
            }
        )

    leaderboard_df = pd.DataFrame(rows)
    if leaderboard_df.empty:
        return leaderboard_df

    leaderboard_df = leaderboard_df.sort_values(
        by=["cost_score", "cost_total_net_pnl"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    leaderboard_df.insert(0, "rank", range(1, len(leaderboard_df.index) + 1))
    return leaderboard_df


def render_sweep_summary(
    leaderboard_df: pd.DataFrame,
    evaluated_no_cost_count: int,
) -> str:
    """Render sweep summary in Chinese."""

    if leaderboard_df.empty:
        return "# Alpha Sweep 总结\n\n没有可用候选结果。"

    best_row = leaderboard_df.iloc[0]
    positive_cost_rows = leaderboard_df[
        (leaderboard_df["cost_bankrupt"] == False)
        & (leaderboard_df["cost_statistics_valid"] == True)
        & (pd.to_numeric(leaderboard_df["cost_total_net_pnl"], errors="coerce") > 0)
    ]
    positive_no_cost_rows = leaderboard_df[
        (leaderboard_df["no_cost_bankrupt"] == False)
        & (leaderboard_df["no_cost_statistics_valid"] == True)
        & (pd.to_numeric(leaderboard_df["no_cost_total_net_pnl"], errors="coerce") > 0)
    ]

    if positive_no_cost_rows.empty:
        recommendation = (
            "已评估的无成本 top candidates 仍未转正，应先修改 alpha 或继续加强入场过滤，"
            "不应进入 live runner，也不应进入正式第 6B 参数优化。"
        )
    elif positive_cost_rows.empty:
        recommendation = (
            "已有候选在无成本下转正，但成本版仍亏损，说明存在弱毛 alpha；"
            "下一步应继续降频、降低交易密度和成本拖累，再考虑进入第 6B。"
        )
    else:
        recommendation = (
            "至少有一个候选在成本版也转正，可以进入正式第 6B 训练/验证参数优化；"
            "仍然不进入 live runner。"
        )

    return (
        "# Alpha Sweep 总结\n\n"
        f"- 已运行成本版候选数：{len(leaderboard_df.index)}\n"
        f"- 已运行无成本复核候选数：{evaluated_no_cost_count}\n"
        f"- 最优成本版候选：{best_row['name']}\n"
        f"- 最优成本版 total_net_pnl：{best_row['cost_total_net_pnl']}\n"
        f"- 最优成本版 sharpe_ratio：{best_row['cost_sharpe_ratio']}\n"
        f"- 最优成本版 max_ddpercent：{best_row['cost_max_ddpercent']}\n"
        f"- 最优成本版 engine_trade_count：{best_row['cost_engine_trade_count']}\n"
        f"- 结论：{recommendation}\n"
    )


def run_sweep(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    """Run the full guarded alpha sweep."""

    base_config_path = resolve_path(args.base_config)
    base_config = load_json_file(base_config_path)
    base_setting = base_config.get("setting")
    if not isinstance(base_setting, dict):
        raise AlphaSweepError(f"base-config 缺少 setting 对象: {base_config_path}")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_runs(base_setting=base_setting, output_dir=output_dir, max_runs=args.max_runs)
    if not candidates:
        raise AlphaSweepError("没有生成任何候选参数")

    write_json(
        output_dir / "sweep_config.json",
        {
            "base_config_path": base_config_path,
            "vt_symbol": args.vt_symbol,
            "start": args.start,
            "end": args.end,
            "timezone": args.timezone,
            "capital": args.capital,
            "rate": args.rate,
            "slippage_mode": args.slippage_mode,
            "slippage": args.slippage,
            "max_runs": args.max_runs,
            "no_cost_also": args.no_cost_also,
            "candidate_names": [candidate.name for candidate in candidates],
        },
    )

    run_cost_backtests(args=args, base_config_path=base_config_path, candidates=candidates, logger=logger)

    top_no_cost_candidates: list[CandidateRun] = []
    if args.no_cost_also:
        top_no_cost_candidates = select_top_cost_candidates(candidates, top_count=min(NO_COST_TOP_COUNT, len(candidates)))
        run_no_cost_backtests(args=args, base_config_path=base_config_path, candidates=top_no_cost_candidates, logger=logger)

    leaderboard_df = build_leaderboard(candidates)
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False, encoding="utf-8")

    top_candidates_payload = leaderboard_df.head(min(5, len(leaderboard_df.index))).to_dict(orient="records")
    write_json(output_dir / "top_candidates.json", top_candidates_payload)
    (output_dir / "sweep_summary.md").write_text(
        render_sweep_summary(leaderboard_df, evaluated_no_cost_count=len(top_no_cost_candidates)),
        encoding="utf-8",
    )

    for candidate in candidates:
        write_json(candidate.candidate_dir / "candidate_summary.json", summarize_candidate(candidate))

    valid_positive_cost = leaderboard_df[
        (leaderboard_df["cost_bankrupt"] == False)
        & (leaderboard_df["cost_statistics_valid"] == True)
        & (pd.to_numeric(leaderboard_df["cost_total_net_pnl"], errors="coerce") > 0)
    ]
    valid_positive_no_cost = leaderboard_df[
        (leaderboard_df["no_cost_bankrupt"] == False)
        & (leaderboard_df["no_cost_statistics_valid"] == True)
        & (pd.to_numeric(leaderboard_df["no_cost_total_net_pnl"], errors="coerce") > 0)
    ]

    summary = {
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "evaluated_no_cost_count": len(top_no_cost_candidates),
        "best_candidate": top_candidates_payload[0] if top_candidates_payload else None,
        "positive_cost_candidate_count": int(len(valid_positive_cost.index)),
        "positive_no_cost_candidate_count": int(len(valid_positive_no_cost.index)),
        "leaderboard_csv": str(leaderboard_path),
        "top_candidates_json": str(output_dir / "top_candidates.json"),
        "sweep_summary_md": str(output_dir / "sweep_summary.md"),
    }
    write_json(output_dir / "sweep_summary.json", summary)

    log_event(
        logger,
        logging.INFO,
        "alpha_sweep.completed",
        "Alpha sweep completed",
        output_dir=output_dir,
        candidate_count=len(candidates),
        evaluated_no_cost_count=len(top_no_cost_candidates),
        positive_cost_candidate_count=summary["positive_cost_candidate_count"],
        positive_no_cost_candidate_count=summary["positive_no_cost_candidate_count"],
    )
    return summary


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("run_alpha_sweep", verbose=args.verbose)

    try:
        summary = run_sweep(args=args, logger=logger)
        print_json_block("Alpha sweep summary:", summary)
        return 0
    except AlphaSweepError as exc:
        log_event(
            logger,
            logging.ERROR,
            "alpha_sweep.error",
            str(exc),
        )
        return 1
    except Exception:
        logger.exception(
            "Unexpected error during alpha sweep",
            extra={"event": "alpha_sweep.unexpected_error"},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
