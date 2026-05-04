#!/usr/bin/env python3
"""Summarize exported backtest reports with bankrupt-focused diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging


class ReportAnalysisError(Exception):
    """Raised when the report directory is incomplete or invalid."""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Analyze one exported backtest report directory.")
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Backtest report directory. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the analysis as one JSON object.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def resolve_report_dir(report_dir_arg: str) -> Path:
    """Resolve report directory relative to project root."""

    path = Path(report_dir_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    if not path.exists():
        raise ReportAnalysisError(f"报告文件不存在: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReportAnalysisError(f"读取 JSON 失败: {path} | {exc!r}") from exc

    if not isinstance(payload, dict):
        raise ReportAnalysisError(f"JSON 顶层结构必须是对象: {path}")
    return payload


def load_csv_file(path: Path) -> pd.DataFrame:
    """Load a CSV file from disk."""

    if not path.exists():
        raise ReportAnalysisError(f"报告文件不存在: {path}")

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ReportAnalysisError(f"读取 CSV 失败: {path} | {exc!r}") from exc


def summarize_volume_distribution(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize traded volume distribution."""

    if "volume" not in trades_df.columns:
        return {"missing_columns": ["volume"]}

    volume_series = pd.to_numeric(trades_df["volume"], errors="coerce").dropna()
    if volume_series.empty:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(volume_series.count()),
        "mean": float(volume_series.mean()),
        "min": float(volume_series.min()),
        "p50": float(volume_series.quantile(0.5)),
        "p90": float(volume_series.quantile(0.9)),
        "p95": float(volume_series.quantile(0.95)),
        "max": float(volume_series.max()),
    }


def pick_worst_days(daily_df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    """Return the worst daily pnl rows."""

    if "net_pnl" not in daily_df.columns:
        return []

    sortable_df = daily_df.copy()
    sortable_df["net_pnl"] = pd.to_numeric(sortable_df["net_pnl"], errors="coerce")
    sortable_df = sortable_df.dropna(subset=["net_pnl"]).sort_values("net_pnl", ascending=True).head(limit)

    rows: list[dict[str, Any]] = []
    for _, row in sortable_df.iterrows():
        rows.append(
            {
                "date": row.get("date"),
                "net_pnl": float(row["net_pnl"]),
                "trade_count": float(pd.to_numeric(pd.Series([row.get("trade_count", 0)]), errors="coerce").fillna(0.0).iloc[0]),
                "balance": float(pd.to_numeric(pd.Series([row.get("balance", 0)]), errors="coerce").fillna(0.0).iloc[0]),
            }
        )
    return rows


def build_analysis(report_dir: Path) -> dict[str, Any]:
    """Build one normalized analysis payload from report artifacts."""

    daily_df = load_csv_file(report_dir / "daily_pnl.csv")
    trades_df = load_csv_file(report_dir / "trades.csv")
    orders_df = load_csv_file(report_dir / "orders.csv")
    stats = load_json_file(report_dir / "stats.json")
    diagnostics = load_json_file(report_dir / "diagnostics.json")

    daily_trade_count_sum = diagnostics.get("daily_trade_count_sum")
    daily_row_count = diagnostics.get("daily_row_count") or len(daily_df.index)
    average_daily_trade_count = (
        float(daily_trade_count_sum) / float(daily_row_count)
        if daily_trade_count_sum is not None and daily_row_count
        else 0.0
    )

    return {
        "report_dir": str(report_dir),
        "bankrupt": bool(stats.get("bankrupt", diagnostics.get("bankrupt", False))),
        "statistics_valid": bool(stats.get("statistics_valid", False)),
        "first_bankrupt_date": diagnostics.get("first_bankrupt_date"),
        "final_balance": diagnostics.get("final_balance"),
        "min_balance": diagnostics.get("min_balance"),
        "total_commission": diagnostics.get("total_commission"),
        "total_slippage": diagnostics.get("total_slippage"),
        "total_turnover": diagnostics.get("total_turnover"),
        "engine_trade_count": stats.get("engine_trade_count", len(trades_df.index)),
        "closed_round_trip_count": stats.get("closed_round_trip_count"),
        "order_count": stats.get("order_count", len(orders_df.index)),
        "average_daily_trade_count": average_daily_trade_count,
        "worst_10_days": pick_worst_days(daily_df, limit=10),
        "volume_distribution": summarize_volume_distribution(trades_df),
    }


def main() -> int:
    """Analyze one backtest report directory."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("analyze_backtest_report", verbose=args.verbose)

    try:
        report_dir = resolve_report_dir(args.report_dir)
        analysis = build_analysis(report_dir)

        log_event(
            logger,
            logging.INFO,
            "report.analysis_complete",
            "Backtest report analysis completed",
            report_dir=report_dir,
            bankrupt=analysis["bankrupt"],
            statistics_valid=analysis["statistics_valid"],
        )

        if args.json:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
        else:
            summary = {
                "report_dir": analysis["report_dir"],
                "bankrupt": analysis["bankrupt"],
                "statistics_valid": analysis["statistics_valid"],
                "first_bankrupt_date": analysis["first_bankrupt_date"],
                "final_balance": analysis["final_balance"],
                "min_balance": analysis["min_balance"],
                "total_commission": analysis["total_commission"],
                "total_slippage": analysis["total_slippage"],
                "total_turnover": analysis["total_turnover"],
                "engine_trade_count": analysis["engine_trade_count"],
                "closed_round_trip_count": analysis["closed_round_trip_count"],
                "order_count": analysis["order_count"],
                "average_daily_trade_count": analysis["average_daily_trade_count"],
            }
            print_json_block("Backtest report analysis:", summary)
            print_json_block("Worst 10 days:", analysis["worst_10_days"])
            print_json_block("Volume distribution:", analysis["volume_distribution"])
        return 0
    except ReportAnalysisError as exc:
        log_event(
            logger,
            logging.ERROR,
            "report.analysis_error",
            str(exc),
        )
        return 1
    except Exception:
        logger.exception(
            "Unexpected error during report analysis",
            extra={"event": "report.analysis_unexpected_error"},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
