#!/usr/bin/env python3
"""Analyze alpha quality from exported backtest reports."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable


class AlphaDiagnosticsError(Exception):
    """Raised when a report directory cannot be analyzed."""


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    """Normalized report artifacts loaded from one report directory."""

    report_dir: Path
    stats: dict[str, Any]
    diagnostics: dict[str, Any]
    daily_df: pd.DataFrame
    trades_df: pd.DataFrame
    orders_df: pd.DataFrame | None
    run_config: dict[str, Any]


@dataclass(slots=True)
class OpenLot:
    """One open trade lot waiting to be matched by a closing fill."""

    side: str
    entry_datetime: pd.Timestamp
    entry_price: float
    remaining_volume: float
    entry_trade_id: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Analyze alpha diagnostics from one backtest report directory.")
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Primary report directory. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--compare-report-dir",
        help="Optional comparison report directory, typically the no-cost report.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Default: <report-dir>/alpha_diagnostics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print alpha_summary.json to stdout after generation.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N rows for worst days and busiest days. Default: 20.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str | None, default_path: Path | None = None) -> Path:
    """Resolve a path argument relative to project root."""

    if path_arg:
        path = Path(path_arg)
    elif default_path is not None:
        path = default_path
    else:
        raise AlphaDiagnosticsError("缺少路径参数且没有默认值")

    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json_file(path: Path, required: bool = True) -> dict[str, Any]:
    """Load one JSON object from disk."""

    if not path.exists():
        if required:
            raise AlphaDiagnosticsError(f"报告文件不存在: {path}")
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AlphaDiagnosticsError(f"读取 JSON 失败: {path} | {exc!r}") from exc

    if not isinstance(payload, dict):
        raise AlphaDiagnosticsError(f"JSON 顶层结构必须是对象: {path}")
    return payload


def load_csv_file(path: Path, required: bool = True) -> pd.DataFrame:
    """Load one CSV file from disk."""

    if not path.exists():
        if required:
            raise AlphaDiagnosticsError(f"报告文件不存在: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise AlphaDiagnosticsError(f"读取 CSV 失败: {path} | {exc!r}") from exc


def load_report_artifacts(report_dir: Path) -> ReportArtifacts:
    """Load all required artifacts from one report directory."""

    if not report_dir.exists():
        raise AlphaDiagnosticsError(f"报告目录不存在: {report_dir}")

    return ReportArtifacts(
        report_dir=report_dir,
        stats=load_json_file(report_dir / "stats.json", required=True),
        diagnostics=load_json_file(report_dir / "diagnostics.json", required=False),
        daily_df=load_csv_file(report_dir / "daily_pnl.csv", required=True),
        trades_df=load_csv_file(report_dir / "trades.csv", required=True),
        orders_df=load_csv_file(report_dir / "orders.csv", required=False),
        run_config=load_json_file(report_dir / "run_config.json", required=True),
    )


def prepare_daily_dataframe(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily pnl output for calendar aggregation."""

    normalized_df = daily_df.copy()
    if "date" not in normalized_df.columns:
        raise AlphaDiagnosticsError("daily_pnl.csv 缺少 date 列，无法做日历分析")

    normalized_df["date"] = pd.to_datetime(normalized_df["date"], errors="coerce")
    normalized_df = normalized_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_columns = ["net_pnl", "trade_count", "balance", "commission", "slippage", "turnover"]
    for column in numeric_columns:
        if column in normalized_df.columns:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce").fillna(0.0)

    return normalized_df


def normalize_direction(value: Any) -> str:
    """Normalize trade direction text to long/short/unknown."""

    text = str(value).strip().lower()
    if text in {"多", "long"}:
        return "long"
    if text in {"空", "short"}:
        return "short"
    return "unknown"


def normalize_offset(value: Any) -> str:
    """Normalize trade offset text to open/close/unknown."""

    text = str(value).strip().lower()
    if text in {"开", "open"}:
        return "open"
    if text in {"平", "close"}:
        return "close"
    return "unknown"


def get_contract_size(run_config: dict[str, Any]) -> float:
    """Read contract size from run_config safely."""

    strategy_setting = run_config.get("strategy_setting", {})
    if isinstance(strategy_setting, dict):
        try:
            contract_size = float(strategy_setting.get("contract_size", 0.0) or 0.0)
            if contract_size > 0:
                return contract_size
        except (TypeError, ValueError):
            pass

    instrument_meta = run_config.get("instrument_meta", {})
    if isinstance(instrument_meta, dict):
        try:
            contract_size = float(instrument_meta.get("size", 0.0) or 0.0)
            if contract_size > 0:
                return contract_size
        except (TypeError, ValueError):
            pass

    raise AlphaDiagnosticsError("run_config.json 缺少可用 contract_size")


def get_absolute_slippage(run_config: dict[str, Any]) -> float:
    """Read absolute slippage from run_config safely."""

    try:
        return float(run_config.get("absolute_slippage", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def get_rate(run_config: dict[str, Any]) -> float:
    """Read commission rate from run_config safely."""

    try:
        return float(run_config.get("rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def pair_round_trips(
    trades_df: pd.DataFrame,
    contract_size: float,
    rate: float,
    absolute_slippage: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Pair open/close fills into round-trip trades for side/hour/duration analysis."""

    warnings: list[str] = []
    if trades_df.empty:
        warnings.append("trades.csv 为空，无法生成 round-trip 级别分析")
        return pd.DataFrame(), warnings

    normalized_df = trades_df.copy()
    normalized_df["datetime"] = pd.to_datetime(normalized_df["datetime"], errors="coerce")
    normalized_df = normalized_df.dropna(subset=["datetime"]).sort_values(
        by=["datetime", "tradeid", "orderid", "vt_tradeid"],
        kind="stable",
    )

    if normalized_df.empty:
        warnings.append("trades.csv 没有可解析 datetime，无法生成 round-trip 级别分析")
        return pd.DataFrame(), warnings

    if normalized_df["datetime"].dt.tz is None:
        normalized_df["datetime"] = normalized_df["datetime"].dt.tz_localize("Asia/Shanghai")
    else:
        normalized_df["datetime"] = normalized_df["datetime"].dt.tz_convert("Asia/Shanghai")

    open_lots: dict[str, list[OpenLot]] = {"long": [], "short": []}
    round_trip_records: list[dict[str, Any]] = []
    epsilon = 1e-12

    for row in normalized_df.to_dict(orient="records"):
        direction = normalize_direction(row.get("direction"))
        offset = normalize_offset(row.get("offset"))
        price = float(row.get("price", 0.0) or 0.0)
        volume = float(row.get("volume", 0.0) or 0.0)
        trade_dt = pd.Timestamp(row["datetime"])
        trade_id = str(row.get("vt_tradeid") or row.get("tradeid") or "")

        if direction == "unknown" or offset == "unknown" or price <= 0 or volume <= 0:
            warnings.append(f"忽略无法识别的成交记录: trade_id={trade_id}")
            continue

        if offset == "open":
            open_lots[direction].append(
                OpenLot(
                    side=direction,
                    entry_datetime=trade_dt,
                    entry_price=price,
                    remaining_volume=volume,
                    entry_trade_id=trade_id,
                )
            )
            continue

        side_to_close = "long" if direction == "short" else "short"
        remaining_close_volume = volume
        queue = open_lots[side_to_close]

        while remaining_close_volume > epsilon and queue:
            lot = queue[0]
            matched_volume = min(remaining_close_volume, lot.remaining_volume)

            if side_to_close == "long":
                gross_pnl = (price - lot.entry_price) * matched_volume * contract_size
            else:
                gross_pnl = (lot.entry_price - price) * matched_volume * contract_size

            entry_turnover = lot.entry_price * matched_volume * contract_size
            exit_turnover = price * matched_volume * contract_size
            commission = (entry_turnover + exit_turnover) * rate
            slippage_cost = matched_volume * contract_size * absolute_slippage * 2.0
            duration_minutes = (trade_dt - lot.entry_datetime).total_seconds() / 60.0

            round_trip_records.append(
                {
                    "side": side_to_close,
                    "entry_datetime": lot.entry_datetime,
                    "exit_datetime": trade_dt,
                    "entry_hour": int(lot.entry_datetime.hour),
                    "entry_price": lot.entry_price,
                    "exit_price": price,
                    "volume": matched_volume,
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "slippage": slippage_cost,
                    "net_pnl": gross_pnl - commission - slippage_cost,
                    "duration_minutes": duration_minutes,
                    "entry_trade_id": lot.entry_trade_id,
                    "exit_trade_id": trade_id,
                }
            )

            lot.remaining_volume -= matched_volume
            remaining_close_volume -= matched_volume
            if lot.remaining_volume <= epsilon:
                queue.pop(0)

        if remaining_close_volume > epsilon:
            warnings.append(
                f"存在无法配对的平仓成交: trade_id={trade_id}, remaining_close_volume={remaining_close_volume:.8f}"
            )

    remaining_open_volume = sum(lot.remaining_volume for lots in open_lots.values() for lot in lots)
    if remaining_open_volume > epsilon:
        warnings.append(f"存在未平仓的开仓成交未能配对: remaining_open_volume={remaining_open_volume:.8f}")

    round_trip_df = pd.DataFrame(round_trip_records)
    if not round_trip_df.empty:
        round_trip_df = round_trip_df.sort_values("entry_datetime").reset_index(drop=True)

    return round_trip_df, warnings


def summarize_monthly_pnl(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily pnl into monthly buckets."""

    summary_df = daily_df.copy()
    summary_df["month"] = summary_df["date"].dt.to_period("M").astype(str)
    monthly_df = (
        summary_df.groupby("month", as_index=False)
        .agg(
            net_pnl=("net_pnl", "sum"),
            trade_count=("trade_count", "sum"),
            win_day_count=("net_pnl", lambda series: int((series > 0).sum())),
            loss_day_count=("net_pnl", lambda series: int((series < 0).sum())),
            avg_daily_pnl=("net_pnl", "mean"),
            end_balance=("balance", "last"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    return monthly_df


def summarize_weekly_pnl(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily pnl into weekly buckets."""

    summary_df = daily_df.copy()
    weekly_period = summary_df["date"].dt.to_period("W-SUN")
    summary_df["week"] = weekly_period.astype(str)
    summary_df["week_start"] = weekly_period.apply(lambda period: period.start_time.date().isoformat())
    summary_df["week_end"] = weekly_period.apply(lambda period: period.end_time.date().isoformat())

    weekly_df = (
        summary_df.groupby(["week", "week_start", "week_end"], as_index=False)
        .agg(
            net_pnl=("net_pnl", "sum"),
            trade_count=("trade_count", "sum"),
            win_day_count=("net_pnl", lambda series: int((series > 0).sum())),
            loss_day_count=("net_pnl", lambda series: int((series < 0).sum())),
            avg_daily_pnl=("net_pnl", "mean"),
            end_balance=("balance", "last"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    return weekly_df


def summarize_trade_side(round_trip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate round-trip pnl by long/short side."""

    if round_trip_df.empty:
        return pd.DataFrame(
            columns=["side", "trade_count", "gross_pnl", "net_pnl", "win_rate", "avg_pnl", "avg_duration_minutes"]
        )

    summary_df = (
        round_trip_df.groupby("side", as_index=False)
        .agg(
            trade_count=("net_pnl", "count"),
            gross_pnl=("gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            win_count=("net_pnl", lambda series: int((series > 0).sum())),
            avg_pnl=("net_pnl", "mean"),
            avg_duration_minutes=("duration_minutes", "mean"),
        )
        .sort_values("side")
        .reset_index(drop=True)
    )
    summary_df["win_rate"] = summary_df["win_count"] / summary_df["trade_count"] * 100.0
    return summary_df.drop(columns=["win_count"])


def summarize_trade_hour(round_trip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate round-trip pnl by Shanghai entry hour."""

    if round_trip_df.empty:
        return pd.DataFrame(columns=["entry_hour", "trade_count", "gross_pnl", "net_pnl", "win_rate", "avg_pnl"])

    summary_df = (
        round_trip_df.groupby("entry_hour", as_index=False)
        .agg(
            trade_count=("net_pnl", "count"),
            gross_pnl=("gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            win_count=("net_pnl", lambda series: int((series > 0).sum())),
            avg_pnl=("net_pnl", "mean"),
        )
        .sort_values("entry_hour")
        .reset_index(drop=True)
    )
    summary_df["win_rate"] = summary_df["win_count"] / summary_df["trade_count"] * 100.0
    return summary_df.drop(columns=["win_count"])


def summarize_trade_duration(round_trip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate round-trip holding duration."""

    if round_trip_df.empty:
        return pd.DataFrame(
            columns=["group", "trade_count", "avg_duration_minutes", "median_duration_minutes", "p90_duration_minutes", "max_duration_minutes"]
        )

    rows: list[dict[str, Any]] = []
    for group_name, group_df in [("all", round_trip_df)] + list(round_trip_df.groupby("side")):
        duration_series = pd.to_numeric(group_df["duration_minutes"], errors="coerce").dropna()
        rows.append(
            {
                "group": group_name,
                "trade_count": int(duration_series.count()),
                "avg_duration_minutes": float(duration_series.mean()) if not duration_series.empty else 0.0,
                "median_duration_minutes": float(duration_series.median()) if not duration_series.empty else 0.0,
                "p90_duration_minutes": float(duration_series.quantile(0.9)) if not duration_series.empty else 0.0,
                "max_duration_minutes": float(duration_series.max()) if not duration_series.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_cost_impact_summary(
    current_artifacts: ReportArtifacts,
    compare_artifacts: ReportArtifacts,
) -> dict[str, Any]:
    """Compare cost and no-cost reports."""

    current_pnl = float(current_artifacts.stats.get("total_net_pnl", 0.0) or 0.0)
    compare_pnl = float(compare_artifacts.stats.get("total_net_pnl", 0.0) or 0.0)
    cost_drag = compare_pnl - current_pnl
    engine_trade_count = max(int(current_artifacts.stats.get("engine_trade_count", 0) or 0), 1)
    round_trip_count = max(int(current_artifacts.stats.get("closed_round_trip_count", 0) or 0), 1)
    compare_abs_gross = abs(float(compare_artifacts.stats.get("gross_profit", 0.0) or 0.0)) + abs(
        float(compare_artifacts.stats.get("gross_loss", 0.0) or 0.0)
    )

    return {
        "current_report_dir": str(current_artifacts.report_dir),
        "compare_report_dir": str(compare_artifacts.report_dir),
        "current_total_net_pnl": current_pnl,
        "compare_total_net_pnl": compare_pnl,
        "total_net_pnl_difference": cost_drag,
        "cost_drag": cost_drag,
        "cost_drag_per_engine_trade": cost_drag / engine_trade_count,
        "cost_drag_per_round_trip": cost_drag / round_trip_count,
        "cost_drag_pct_of_abs_gross": (cost_drag / compare_abs_gross * 100.0) if compare_abs_gross > 0 else None,
        "current_engine_trade_count": engine_trade_count,
        "current_closed_round_trip_count": round_trip_count,
        "compare_is_no_cost": is_no_cost_report(compare_artifacts),
    }


def is_no_cost_report(artifacts: ReportArtifacts) -> bool:
    """Return whether one report is clearly a no-cost run."""

    return get_rate(artifacts.run_config) == 0.0 and get_absolute_slippage(artifacts.run_config) == 0.0


def determine_alpha_status(
    current_artifacts: ReportArtifacts,
    compare_artifacts: ReportArtifacts | None,
) -> tuple[bool | None, str, str]:
    """Classify whether the strategy has gross alpha and what the main issue is."""

    current_pnl = float(current_artifacts.stats.get("total_net_pnl", 0.0) or 0.0)
    if compare_artifacts and is_no_cost_report(compare_artifacts):
        compare_pnl = float(compare_artifacts.stats.get("total_net_pnl", 0.0) or 0.0)
        if compare_pnl <= 0:
            return (
                False,
                "当前无成本版本仍为负收益，说明策略没有可验证的毛 alpha，成本只是进一步放大亏损。",
                "alpha_negative_and_cost_negative",
            )
        if current_pnl <= 0:
            return (
                True,
                "存在一定毛 alpha，但成本版仍为负，说明当前 alpha 太弱，无法覆盖手续费与滑点。",
                "weak_alpha_cost_drag_heavy",
            )
        return (
            True,
            "无成本与有成本版本均为正，说明存在成本后仍可存活的 alpha，但仍需进入更严格的 6B 训练/验证而非 live。",
            "cost_adjusted_alpha_positive",
        )

    if is_no_cost_report(current_artifacts):
        if current_pnl <= 0:
            return (
                False,
                "当前报告本身就是无成本版本且仍为负收益，说明没有毛 alpha。",
                "alpha_negative_without_compare",
            )
        return (
            True,
            "当前报告是无成本版本且收益为正，说明存在毛 alpha，但仍需要成本版验证。",
            "alpha_positive_without_compare",
        )

    return (
        None,
        "当前没有可用无成本对照，只能确认成本版表现，无法单独证明毛 alpha。",
        "unknown_without_no_cost_reference",
    )


def build_priority_recommendations(
    current_artifacts: ReportArtifacts,
    trade_hour_df: pd.DataFrame,
) -> list[str]:
    """Create pragmatic next-step filter recommendations."""

    recommendations: list[str] = [
        "优先继续做降频过滤，不要通过扩大仓位改善结果。",
        "先提高 breakout_window 或增加 cooldown_bars，降低重复入场密度。",
        "启用或收紧 ATR 入场过滤、breakout_atr 过滤和 regime persistence 过滤。",
        "缩窄 5m 波动区间过滤，避免在过低或过高波动下都频繁交易。",
    ]

    avg_engine_trades_per_day = float(current_artifacts.stats.get("engine_trade_count", 0) or 0) / max(
        int(len(current_artifacts.daily_df.index)),
        1,
    )
    if avg_engine_trades_per_day > 15:
        recommendations.append("当前成交密度偏高，优先减少日均成交次数，再看毛 alpha 是否改善。")

    if not trade_hour_df.empty:
        worst_hour = trade_hour_df.sort_values("net_pnl", ascending=True).iloc[0]
        recommendations.append(
            f"优先检查上海时间 {int(worst_hour['entry_hour']):02d}:00 时段，当前该小时段 round-trip net_pnl 最差。"
        )

    return recommendations


def build_alpha_summary(
    current_artifacts: ReportArtifacts,
    daily_df: pd.DataFrame,
    round_trip_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    trade_side_df: pd.DataFrame,
    trade_hour_df: pd.DataFrame,
    trade_duration_df: pd.DataFrame,
    top_n: int,
    compare_artifacts: ReportArtifacts | None = None,
    pairing_warnings: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build one alpha diagnostics summary payload."""

    pairing_warnings = pairing_warnings or []
    cost_impact_summary = (
        build_cost_impact_summary(current_artifacts, compare_artifacts) if compare_artifacts else None
    )
    has_gross_alpha, alpha_status_text, alpha_status_code = determine_alpha_status(current_artifacts, compare_artifacts)
    priority_recommendations = build_priority_recommendations(current_artifacts, trade_hour_df)

    worst_days = daily_df.sort_values("net_pnl", ascending=True).head(top_n)
    busiest_days = daily_df.sort_values("trade_count", ascending=False).head(top_n)
    avg_engine_trades_per_day = float(current_artifacts.stats.get("engine_trade_count", 0) or 0) / max(
        int(len(daily_df.index)),
        1,
    )
    avg_round_trips_per_day = float(current_artifacts.stats.get("closed_round_trip_count", 0) or 0) / max(
        int(len(daily_df.index)),
        1,
    )

    if compare_artifacts and is_no_cost_report(compare_artifacts):
        if has_gross_alpha is False:
            should_enter_formal_optimization = False
            should_pause_live_runner = True
        elif has_gross_alpha is True and float(current_artifacts.stats.get("total_net_pnl", 0.0) or 0.0) <= 0:
            should_enter_formal_optimization = False
            should_pause_live_runner = True
        else:
            should_enter_formal_optimization = True
            should_pause_live_runner = True
    else:
        should_enter_formal_optimization = False
        should_pause_live_runner = True

    summary: dict[str, Any] = {
        "report_dir": str(current_artifacts.report_dir),
        "compare_report_dir": str(compare_artifacts.report_dir) if compare_artifacts else None,
        "has_gross_alpha": has_gross_alpha,
        "alpha_status_code": alpha_status_code,
        "alpha_status_text": alpha_status_text,
        "current_bankrupt": bool(current_artifacts.stats.get("bankrupt", False)),
        "current_statistics_valid": bool(current_artifacts.stats.get("statistics_valid", False)),
        "current_total_net_pnl": float(current_artifacts.stats.get("total_net_pnl", 0.0) or 0.0),
        "current_final_balance": float(current_artifacts.stats.get("final_balance", 0.0) or 0.0),
        "current_max_ddpercent": float(current_artifacts.stats.get("max_ddpercent", 0.0) or 0.0),
        "current_sharpe_ratio": float(current_artifacts.stats.get("sharpe_ratio", 0.0) or 0.0),
        "current_engine_trade_count": int(current_artifacts.stats.get("engine_trade_count", 0) or 0),
        "current_closed_round_trip_count": int(current_artifacts.stats.get("closed_round_trip_count", 0) or 0),
        "avg_engine_trades_per_day": avg_engine_trades_per_day,
        "avg_round_trips_per_day": avg_round_trips_per_day,
        "worst_hours": trade_hour_df.sort_values("net_pnl", ascending=True).head(min(top_n, 5)).to_dict(orient="records")
        if not trade_hour_df.empty
        else [],
        "worst_days": worst_days[["date", "net_pnl", "trade_count", "balance"]].to_dict(orient="records"),
        "busiest_days": busiest_days[["date", "net_pnl", "trade_count", "balance"]].to_dict(orient="records"),
        "trade_side_summary": trade_side_df.to_dict(orient="records"),
        "trade_duration_summary": trade_duration_df.to_dict(orient="records"),
        "monthly_overview": monthly_df.to_dict(orient="records"),
        "weekly_overview": weekly_df.to_dict(orient="records"),
        "cost_impact_summary": cost_impact_summary,
        "pairing_warnings": pairing_warnings,
        "priority_recommendations": priority_recommendations,
        "should_enter_formal_parameter_optimization": should_enter_formal_optimization,
        "should_pause_live_runner": should_pause_live_runner,
    }
    return summary, cost_impact_summary


def render_markdown(summary: dict[str, Any]) -> str:
    """Render alpha diagnostics markdown in Chinese."""

    has_gross_alpha = summary.get("has_gross_alpha")
    if has_gross_alpha is False:
        gross_alpha_line = "当前策略没有可验证的毛 alpha。无成本版本仍为负收益，不能把这解释成策略有效。"
    elif has_gross_alpha is True:
        gross_alpha_line = "当前策略存在一定毛 alpha，但是否足以覆盖成本，需要看成本版是否同步转正。"
    else:
        gross_alpha_line = "当前缺少明确的无成本对照，暂时无法单独证明毛 alpha。"

    if summary.get("should_enter_formal_parameter_optimization"):
        optimization_line = "可以进入正式第 6B 训练/验证参数优化，但仍不进入 live runner。"
    else:
        optimization_line = "当前不应进入正式第 6B 参数优化，应先继续做 alpha 诊断和降频过滤。"

    if summary.get("should_pause_live_runner", True):
        live_line = "应暂缓 live runner。当前证据不足以支持上线。"
    else:
        live_line = "不建议直接进入 live runner，至少还需要 6B 验证。"

    cost_impact_summary = summary.get("cost_impact_summary") or {}
    cost_line = ""
    if cost_impact_summary:
        cost_line = (
            f"成本拖累为 {cost_impact_summary.get('cost_drag'):.6f}，"
            f"单笔 engine trade 平均拖累 {cost_impact_summary.get('cost_drag_per_engine_trade'):.6f}。"
        )

    recommendations = summary.get("priority_recommendations", [])
    recommendation_lines = "\n".join(f"- {item}" for item in recommendations) or "- 暂无"

    warning_lines = "\n".join(f"- {item}" for item in summary.get("pairing_warnings", [])) or "- 无"

    return (
        "# Alpha 诊断报告\n\n"
        "## 结论\n"
        f"- 毛 alpha 判断：{gross_alpha_line}\n"
        f"- 当前主要亏损来源：{summary.get('alpha_status_text')}\n"
        f"- 是否应该进入参数优化：{optimization_line}\n"
        f"- 是否应该暂缓 live runner：{live_line}\n"
        f"- 成本影响：{cost_line or '未提供无成本对照，暂无法单独量化成本拖累。'}\n\n"
        "## 关键指标\n"
        f"- 成本版 total_net_pnl：{summary.get('current_total_net_pnl'):.6f}\n"
        f"- 成本版 max_ddpercent：{summary.get('current_max_ddpercent'):.6f}\n"
        f"- 成本版 sharpe_ratio：{summary.get('current_sharpe_ratio'):.6f}\n"
        f"- 平均每天 engine_trade_count：{summary.get('avg_engine_trades_per_day'):.6f}\n"
        f"- 平均每天 round_trip：{summary.get('avg_round_trips_per_day'):.6f}\n\n"
        "## 优先建议过滤器\n"
        f"{recommendation_lines}\n\n"
        "## 配对警告\n"
        f"{warning_lines}\n"
    )


def write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def run_analysis(
    report_dir: Path,
    compare_report_dir: Path | None,
    output_dir: Path,
    top_n: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run the full alpha diagnostics workflow."""

    current_artifacts = load_report_artifacts(report_dir)
    compare_artifacts = load_report_artifacts(compare_report_dir) if compare_report_dir else None

    daily_df = prepare_daily_dataframe(current_artifacts.daily_df)
    contract_size = get_contract_size(current_artifacts.run_config)
    rate = get_rate(current_artifacts.run_config)
    absolute_slippage = get_absolute_slippage(current_artifacts.run_config)

    round_trip_df, pairing_warnings = pair_round_trips(
        current_artifacts.trades_df,
        contract_size=contract_size,
        rate=rate,
        absolute_slippage=absolute_slippage,
    )

    monthly_df = summarize_monthly_pnl(daily_df)
    weekly_df = summarize_weekly_pnl(daily_df)
    trade_side_df = summarize_trade_side(round_trip_df)
    trade_hour_df = summarize_trade_hour(round_trip_df)
    trade_duration_df = summarize_trade_duration(round_trip_df)
    daily_worst_df = daily_df.sort_values("net_pnl", ascending=True).head(top_n).reset_index(drop=True)

    alpha_summary, cost_impact_summary = build_alpha_summary(
        current_artifacts=current_artifacts,
        daily_df=daily_df,
        round_trip_df=round_trip_df,
        monthly_df=monthly_df,
        weekly_df=weekly_df,
        trade_side_df=trade_side_df,
        trade_hour_df=trade_hour_df,
        trade_duration_df=trade_duration_df,
        top_n=top_n,
        compare_artifacts=compare_artifacts,
        pairing_warnings=pairing_warnings,
    )

    alpha_markdown = render_markdown(alpha_summary)

    write_json(output_dir / "alpha_summary.json", alpha_summary)
    write_dataframe_csv(monthly_df, output_dir / "monthly_pnl.csv")
    write_dataframe_csv(weekly_df, output_dir / "weekly_pnl.csv")
    write_dataframe_csv(daily_worst_df, output_dir / "daily_worst.csv")
    write_dataframe_csv(trade_side_df, output_dir / "trade_side_summary.csv")
    write_dataframe_csv(trade_hour_df, output_dir / "trade_hour_summary.csv")
    write_dataframe_csv(trade_duration_df, output_dir / "trade_duration_summary.csv")
    if cost_impact_summary is not None:
        write_json(output_dir / "cost_impact_summary.json", cost_impact_summary)
    (output_dir / "alpha_diagnostics.md").write_text(alpha_markdown, encoding="utf-8")

    log_event(
        logger,
        logging.INFO,
        "alpha_diagnostics.completed",
        "Alpha diagnostics analysis completed",
        report_dir=report_dir,
        compare_report_dir=compare_report_dir,
        output_dir=output_dir,
        has_gross_alpha=alpha_summary.get("has_gross_alpha"),
        current_total_net_pnl=alpha_summary.get("current_total_net_pnl"),
    )

    return alpha_summary


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("analyze_alpha_diagnostics", verbose=args.verbose)

    try:
        report_dir = resolve_path(args.report_dir)
        compare_report_dir = resolve_path(args.compare_report_dir) if args.compare_report_dir else None
        output_dir = resolve_path(args.output_dir, default_path=report_dir / "alpha_diagnostics")
        alpha_summary = run_analysis(
            report_dir=report_dir,
            compare_report_dir=compare_report_dir,
            output_dir=output_dir,
            top_n=args.top_n,
            logger=logger,
        )

        if args.json:
            print(json.dumps(to_jsonable(alpha_summary), ensure_ascii=False, indent=2))
        else:
            print_json_block("Alpha diagnostics summary:", alpha_summary)
        return 0
    except AlphaDiagnosticsError as exc:
        log_event(
            logger,
            logging.ERROR,
            "alpha_diagnostics.error",
            str(exc),
        )
        return 1
    except Exception:
        logger.exception(
            "Unexpected error during alpha diagnostics analysis",
            extra={"event": "alpha_diagnostics.unexpected_error"},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
