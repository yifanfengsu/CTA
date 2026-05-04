#!/usr/bin/env python3
"""Analyze trade attribution from exported backtest reports."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable


class TradeAttributionError(Exception):
    """Raised when trade attribution cannot continue."""


@dataclass(slots=True)
class ReportArtifacts:
    """Backtest report artifacts loaded from disk."""

    report_dir: Path
    stats: dict[str, Any]
    diagnostics: dict[str, Any]
    trades_df: pd.DataFrame
    orders_df: pd.DataFrame
    daily_df: pd.DataFrame
    warnings: list[str]


@dataclass(slots=True)
class OpenLot:
    """One open fill waiting to be matched by a closing fill."""

    side: str
    entry_datetime: pd.Timestamp
    entry_price: float
    remaining_volume: float
    entry_trade_id: str
    entry_order_id: str


EXIT_REASON_COLUMNS: tuple[str, ...] = (
    "exit_reason",
    "close_reason",
    "stop_reason",
    "reason",
    "signal",
    "signal_name",
    "strategy_signal",
    "strategy_reason",
    "reference",
)

GROUP_STAT_COLUMNS: list[str] = [
    "trade_count",
    "gross_pnl",
    "net_pnl",
    "win_count",
    "loss_count",
    "win_rate_pct",
    "average_pnl",
    "median_pnl",
    "average_win",
    "average_loss",
    "profit_loss_ratio",
    "expectancy",
    "max_win",
    "max_loss",
    "total_volume",
    "avg_duration_minutes",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Analyze trade attribution from one backtest report directory.")
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Backtest report directory. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Default: <report-dir>/trade_attribution.",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="Timezone used for hour/weekday attribution. Default: Asia/Shanghai.",
    )
    parser.add_argument(
        "--bar-db-check",
        action="store_true",
        help="Reserve flag for future bar database cross-checks. Current implementation records a skipped warning.",
    )
    parser.add_argument(
        "--format",
        action="append",
        default=[],
        help="Output format: json, csv, md. Can be repeated or comma-separated. Default: all formats.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print attribution_summary.json to stdout after generation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str | None, default_path: Path | None = None) -> Path:
    """Resolve path arguments relative to project root."""

    if path_arg:
        path = Path(path_arg)
    elif default_path is not None:
        path = default_path
    else:
        raise TradeAttributionError("缺少路径参数且没有默认值")

    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def normalize_formats(raw_formats: list[str]) -> set[str]:
    """Normalize output format arguments."""

    if not raw_formats:
        return {"json", "csv", "md"}

    formats: set[str] = set()
    for raw_item in raw_formats:
        for item in str(raw_item).split(","):
            normalized = item.strip().lower()
            if not normalized:
                continue
            if normalized not in {"json", "csv", "md"}:
                raise TradeAttributionError(f"--format 只支持 json/csv/md: {normalized}")
            formats.add(normalized)

    if not formats:
        return {"json", "csv", "md"}
    return formats


def number_or_none(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def first_number(*values: Any) -> float | None:
    """Return the first finite numeric value."""

    for value in values:
        number = number_or_none(value)
        if number is not None:
            return number
    return None


def read_json_optional(path: Path, warnings: list[str]) -> dict[str, Any]:
    """Read one JSON object, warning instead of failing when missing."""

    if not path.exists():
        warnings.append(f"缺少文件: {path.name}")
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"读取 JSON 失败: {path.name} | {exc!r}")
        return {}

    if not isinstance(payload, dict):
        warnings.append(f"JSON 顶层结构不是对象: {path.name}")
        return {}
    return payload


def read_csv_optional(path: Path, warnings: list[str]) -> pd.DataFrame:
    """Read one CSV file, warning instead of failing when missing."""

    if not path.exists():
        warnings.append(f"缺少文件: {path.name}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"读取 CSV 失败: {path.name} | {exc!r}")
        return pd.DataFrame()


def load_report_artifacts(report_dir: Path) -> ReportArtifacts:
    """Load report artifacts from one directory."""

    if not report_dir.exists():
        raise TradeAttributionError(f"报告目录不存在: {report_dir}")
    if not report_dir.is_dir():
        raise TradeAttributionError(f"报告路径不是目录: {report_dir}")

    warnings: list[str] = []
    return ReportArtifacts(
        report_dir=report_dir,
        stats=read_json_optional(report_dir / "stats.json", warnings),
        diagnostics=read_json_optional(report_dir / "diagnostics.json", warnings),
        trades_df=read_csv_optional(report_dir / "trades.csv", warnings),
        orders_df=read_csv_optional(report_dir / "orders.csv", warnings),
        daily_df=read_csv_optional(report_dir / "daily_pnl.csv", warnings),
        warnings=warnings,
    )


def normalize_direction(value: Any) -> str:
    """Normalize direction to long/short/unknown."""

    text = str(value).strip().lower()
    if text in {"多", "long", "buy"}:
        return "long"
    if text in {"空", "short", "sell"}:
        return "short"
    return "unknown"


def normalize_offset(value: Any) -> str:
    """Normalize offset to open/close/unknown."""

    text = str(value).strip().lower()
    if text in {"开", "open"}:
        return "open"
    if text in {"平", "close", "平仓"}:
        return "close"
    return "unknown"


def normalize_datetime_series(
    series: pd.Series,
    timezone_name: str,
    warnings: list[str],
    field_name: str,
) -> pd.Series:
    """Parse datetimes and convert/localize to the configured timezone."""

    timezone = ZoneInfo(timezone_name)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.empty:
        return parsed

    try:
        if parsed.dt.tz is None:
            return parsed.dt.tz_localize(timezone)
        return parsed.dt.tz_convert(timezone)
    except Exception:
        try:
            parsed_utc = pd.to_datetime(series, errors="coerce", utc=True)
            return parsed_utc.dt.tz_convert(timezone)
        except Exception as exc:
            warnings.append(f"{field_name} 时间解析失败: {exc!r}")
            return parsed


def prepare_daily_dataframe(daily_df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Normalize daily pnl data."""

    if daily_df.empty:
        return pd.DataFrame(
            columns=["date", "net_pnl", "trade_count", "balance", "commission", "slippage", "turnover"]
        )

    normalized_df = daily_df.copy()
    if "date" not in normalized_df.columns:
        warnings.append("daily_pnl.csv 缺少 date 列，跳过日级归因")
        return pd.DataFrame(
            columns=["date", "net_pnl", "trade_count", "balance", "commission", "slippage", "turnover"]
        )

    normalized_df["date"] = pd.to_datetime(normalized_df["date"], errors="coerce")
    invalid_count = int(normalized_df["date"].isna().sum())
    if invalid_count:
        warnings.append(f"daily_pnl.csv 有 {invalid_count} 行 date 无法解析，已忽略")
    normalized_df = normalized_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for column in ["net_pnl", "trade_count", "balance", "commission", "slippage", "turnover", "drawdown", "ddpercent"]:
        if column in normalized_df.columns:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")
        else:
            normalized_df[column] = 0.0

    return normalized_df


def prepare_trades_dataframe(
    trades_df: pd.DataFrame,
    timezone_name: str,
    warnings: list[str],
) -> pd.DataFrame:
    """Normalize trade fill data."""

    expected_columns = ["datetime", "direction", "offset", "price", "volume"]
    if trades_df.empty:
        return pd.DataFrame(columns=expected_columns)

    normalized_df = trades_df.copy()
    missing_columns = [column for column in expected_columns if column not in normalized_df.columns]
    if missing_columns:
        warnings.append(f"trades.csv 缺少列: {', '.join(missing_columns)}")
        return pd.DataFrame(columns=expected_columns)

    normalized_df["datetime"] = normalize_datetime_series(normalized_df["datetime"], timezone_name, warnings, "trades.datetime")
    normalized_df["price"] = pd.to_numeric(normalized_df["price"], errors="coerce")
    normalized_df["volume"] = pd.to_numeric(normalized_df["volume"], errors="coerce")
    normalized_df["normalized_direction"] = normalized_df["direction"].map(normalize_direction)
    normalized_df["normalized_offset"] = normalized_df["offset"].map(normalize_offset)

    invalid_datetime_count = int(normalized_df["datetime"].isna().sum())
    if invalid_datetime_count:
        warnings.append(f"trades.csv 有 {invalid_datetime_count} 行 datetime 无法解析，已忽略")

    normalized_df = normalized_df.dropna(subset=["datetime", "price", "volume"]).copy()
    normalized_df = normalized_df[(normalized_df["price"] > 0) & (normalized_df["volume"] > 0)]

    sort_columns = [column for column in ["datetime", "tradeid", "orderid", "vt_tradeid", "vt_orderid"] if column in normalized_df.columns]
    if sort_columns:
        normalized_df = normalized_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return normalized_df


def build_order_reason_map(orders_df: pd.DataFrame) -> dict[str, str]:
    """Build an order-id to metadata/reason map from orders.csv when available."""

    if orders_df.empty:
        return {}

    reason_columns = [column for column in EXIT_REASON_COLUMNS if column in orders_df.columns]
    if not reason_columns:
        return {}

    order_reason_map: dict[str, str] = {}
    id_columns = [column for column in ["vt_orderid", "orderid"] if column in orders_df.columns]
    if not id_columns:
        return {}

    for _, row in orders_df.iterrows():
        reason = ""
        for column in reason_columns:
            candidate = str(row.get(column, "")).strip()
            if candidate and candidate.lower() != "nan":
                reason = candidate
                break
        if not reason:
            continue
        for column in id_columns:
            order_id = str(row.get(column, "")).strip()
            if order_id and order_id.lower() != "nan":
                order_reason_map[order_id] = reason
    return order_reason_map


def get_row_identifier(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    """Return the first non-empty identifier from a row."""

    for column in columns:
        value = str(row.get(column, "")).strip()
        if value and value.lower() != "nan":
            return value
    return ""


def get_exit_reason(row: dict[str, Any], order_reason_map: dict[str, str]) -> str | None:
    """Extract exit reason from trade row or matched order metadata."""

    for column in EXIT_REASON_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value and value.lower() != "nan":
            return value

    for column in ("vt_orderid", "orderid"):
        order_id = str(row.get(column, "")).strip()
        if order_id and order_id in order_reason_map:
            return order_reason_map[order_id]
    return None


def pair_round_trips(
    trades_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """Pair trade fills into FIFO round trips."""

    columns = [
        "side",
        "entry_datetime",
        "exit_datetime",
        "entry_hour",
        "entry_weekday",
        "month",
        "entry_price",
        "exit_price",
        "volume",
        "raw_gross_pnl",
        "raw_turnover",
        "duration_minutes",
        "entry_trade_id",
        "exit_trade_id",
        "entry_order_id",
        "exit_order_id",
        "exit_reason",
    ]
    if trades_df.empty:
        warnings.append("trades.csv 为空或不可用，无法推断往返交易")
        return pd.DataFrame(columns=columns)

    open_lots: dict[str, list[OpenLot]] = {"long": [], "short": []}
    order_reason_map = build_order_reason_map(orders_df)
    records: list[dict[str, Any]] = []
    epsilon = 1e-12
    ignored_count = 0
    unmatched_close_count = 0

    for row in trades_df.to_dict(orient="records"):
        direction = normalize_direction(row.get("direction"))
        offset = normalize_offset(row.get("offset"))
        price = float(row.get("price", 0.0) or 0.0)
        volume = float(row.get("volume", 0.0) or 0.0)
        trade_dt = pd.Timestamp(row["datetime"])
        trade_id = get_row_identifier(row, ("vt_tradeid", "tradeid"))
        order_id = get_row_identifier(row, ("vt_orderid", "orderid"))

        if direction == "unknown" or offset == "unknown" or price <= 0 or volume <= 0:
            ignored_count += 1
            continue

        if offset == "open":
            open_lots[direction].append(
                OpenLot(
                    side=direction,
                    entry_datetime=trade_dt,
                    entry_price=price,
                    remaining_volume=volume,
                    entry_trade_id=trade_id,
                    entry_order_id=order_id,
                )
            )
            continue

        side_to_close = "long" if direction == "short" else "short"
        remaining_close_volume = volume
        queue = open_lots[side_to_close]
        exit_reason = get_exit_reason(row, order_reason_map)

        while remaining_close_volume > epsilon and queue:
            lot = queue[0]
            matched_volume = min(remaining_close_volume, lot.remaining_volume)

            if side_to_close == "long":
                raw_gross_pnl = (price - lot.entry_price) * matched_volume
            else:
                raw_gross_pnl = (lot.entry_price - price) * matched_volume

            duration_minutes = (trade_dt - lot.entry_datetime).total_seconds() / 60.0
            records.append(
                {
                    "side": side_to_close,
                    "entry_datetime": lot.entry_datetime,
                    "exit_datetime": trade_dt,
                    "entry_hour": int(lot.entry_datetime.hour),
                    "entry_weekday": int(lot.entry_datetime.weekday()),
                    "month": lot.entry_datetime.strftime("%Y-%m"),
                    "entry_price": lot.entry_price,
                    "exit_price": price,
                    "volume": matched_volume,
                    "raw_gross_pnl": raw_gross_pnl,
                    "raw_turnover": (lot.entry_price + price) * matched_volume,
                    "duration_minutes": duration_minutes,
                    "entry_trade_id": lot.entry_trade_id,
                    "exit_trade_id": trade_id,
                    "entry_order_id": lot.entry_order_id,
                    "exit_order_id": order_id,
                    "exit_reason": exit_reason,
                }
            )

            lot.remaining_volume -= matched_volume
            remaining_close_volume -= matched_volume
            if lot.remaining_volume <= epsilon:
                queue.pop(0)

        if remaining_close_volume > epsilon:
            unmatched_close_count += 1

    if ignored_count:
        warnings.append(f"忽略无法识别的成交记录数: {ignored_count}")
    if unmatched_close_count:
        warnings.append(f"存在无法配对的平仓成交数: {unmatched_close_count}")

    remaining_open_volume = sum(lot.remaining_volume for lots in open_lots.values() for lot in lots)
    if remaining_open_volume > epsilon:
        warnings.append(f"存在未平仓开仓量，无法完整归因: remaining_open_volume={remaining_open_volume:.8f}")

    round_trip_df = pd.DataFrame(records, columns=columns)
    if not round_trip_df.empty:
        round_trip_df = round_trip_df.sort_values(["entry_datetime", "exit_datetime"], kind="stable").reset_index(drop=True)
    return round_trip_df


def extract_total_net_pnl(stats: dict[str, Any], diagnostics: dict[str, Any], daily_df: pd.DataFrame) -> float:
    """Extract total net pnl, preferring manual diagnostics over invalid stats."""

    daily_sum = None
    if not daily_df.empty and "net_pnl" in daily_df.columns:
        daily_sum = float(pd.to_numeric(daily_df["net_pnl"], errors="coerce").fillna(0.0).sum())

    diagnostics_total = number_or_none(diagnostics.get("total_net_pnl_sum"))
    stats_total = number_or_none(stats.get("total_net_pnl"))
    statistics_valid = bool(stats.get("statistics_valid", True))
    bankrupt = bool(stats.get("bankrupt", diagnostics.get("bankrupt", False)))

    if diagnostics_total is not None:
        return diagnostics_total
    if daily_sum is not None:
        return daily_sum
    if stats_total is not None and (statistics_valid or not bankrupt):
        return stats_total
    if stats_total is not None:
        return stats_total
    return 0.0


def extract_total_costs(stats: dict[str, Any], diagnostics: dict[str, Any], daily_df: pd.DataFrame) -> tuple[float, float]:
    """Extract total commission and slippage."""

    daily_commission = None
    daily_slippage = None
    if not daily_df.empty:
        if "commission" in daily_df.columns:
            daily_commission = float(pd.to_numeric(daily_df["commission"], errors="coerce").fillna(0.0).sum())
        if "slippage" in daily_df.columns:
            daily_slippage = float(pd.to_numeric(daily_df["slippage"], errors="coerce").fillna(0.0).sum())

    commission = first_number(diagnostics.get("total_commission"), stats.get("total_commission"), daily_commission) or 0.0
    slippage = first_number(diagnostics.get("total_slippage"), stats.get("total_slippage"), daily_slippage) or 0.0
    return commission, slippage


def infer_contract_multiplier(round_trip_df: pd.DataFrame, stats: dict[str, Any], warnings: list[str]) -> float:
    """Infer the contract multiplier from round-trip pnl and exported gross stats."""

    if round_trip_df.empty:
        return 1.0

    raw_pnls = pd.to_numeric(round_trip_df["raw_gross_pnl"], errors="coerce").fillna(0.0)
    raw_gross_profit = float(raw_pnls[raw_pnls > 0].sum())
    raw_gross_loss = abs(float(raw_pnls[raw_pnls < 0].sum()))
    stats_gross_profit = number_or_none(stats.get("gross_profit"))
    stats_gross_loss = number_or_none(stats.get("gross_loss"))

    ratios: list[float] = []
    if raw_gross_profit > 0 and stats_gross_profit is not None and stats_gross_profit > 0:
        ratios.append(stats_gross_profit / raw_gross_profit)
    if raw_gross_loss > 0 and stats_gross_loss is not None and stats_gross_loss > 0:
        ratios.append(stats_gross_loss / raw_gross_loss)

    ratios = [ratio for ratio in ratios if np.isfinite(ratio) and ratio > 0]
    if not ratios:
        warnings.append("无法从 stats gross_profit/gross_loss 推断合约乘数，交易级 PnL 使用 price_diff * volume")
        return 1.0

    multiplier = float(np.median(ratios))
    if len(ratios) >= 2:
        low = min(ratios)
        high = max(ratios)
        if low > 0 and (high - low) / low > 0.05:
            warnings.append(
                "由 gross_profit/gross_loss 推断的合约乘数不完全一致，"
                f"ratios={','.join(f'{ratio:.8f}' for ratio in ratios)}"
            )
    return multiplier


def apply_pnl_scaling(
    round_trip_df: pd.DataFrame,
    stats: dict[str, Any],
    total_commission: float,
    total_slippage: float,
    warnings: list[str],
) -> pd.DataFrame:
    """Scale raw round-trip pnl and allocate total costs by turnover."""

    if round_trip_df.empty:
        result_df = round_trip_df.copy()
        for column in ["gross_pnl", "turnover", "allocated_commission", "allocated_slippage", "net_pnl"]:
            result_df[column] = []
        return result_df

    result_df = round_trip_df.copy()
    multiplier = infer_contract_multiplier(result_df, stats, warnings)
    result_df["contract_multiplier"] = multiplier
    result_df["gross_pnl"] = pd.to_numeric(result_df["raw_gross_pnl"], errors="coerce").fillna(0.0) * multiplier
    result_df["turnover"] = pd.to_numeric(result_df["raw_turnover"], errors="coerce").fillna(0.0) * multiplier

    total_turnover = float(result_df["turnover"].sum())
    if total_turnover > 0:
        result_df["allocated_commission"] = total_commission * result_df["turnover"] / total_turnover
        result_df["allocated_slippage"] = total_slippage * result_df["turnover"] / total_turnover
    else:
        result_df["allocated_commission"] = 0.0
        result_df["allocated_slippage"] = 0.0

    result_df["net_pnl"] = result_df["gross_pnl"] - result_df["allocated_commission"] - result_df["allocated_slippage"]
    return result_df


def summarize_pnl_series(pnl_series: pd.Series) -> dict[str, Any]:
    """Summarize a pnl series."""

    clean_series = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if clean_series.empty:
        return {
            "count": 0,
            "total_net_pnl": 0.0,
            "average_pnl": None,
            "median_pnl": None,
            "win_count": 0,
            "loss_count": 0,
            "win_rate_pct": None,
            "win_rate": None,
            "average_win": None,
            "average_loss": None,
            "average_loss_abs": None,
            "profit_loss_ratio": None,
            "expectancy": None,
            "max_win": None,
            "max_loss": None,
            "max_consecutive_losses": 0,
        }

    wins = clean_series[clean_series > 0]
    losses = clean_series[clean_series < 0]
    average_win = float(wins.mean()) if not wins.empty else None
    average_loss = float(losses.mean()) if not losses.empty else None
    average_loss_abs = abs(average_loss) if average_loss is not None else None
    profit_loss_ratio = (
        average_win / average_loss_abs
        if average_win is not None and average_loss_abs not in (None, 0.0)
        else None
    )

    max_consecutive_losses = 0
    current_losses = 0
    for pnl in clean_series:
        if pnl < 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0

    win_rate_pct = float(len(wins) / len(clean_series) * 100.0)
    return {
        "count": int(len(clean_series)),
        "total_net_pnl": float(clean_series.sum()),
        "average_pnl": float(clean_series.mean()),
        "median_pnl": float(clean_series.median()),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate_pct": win_rate_pct,
        "win_rate": win_rate_pct,
        "average_win": average_win,
        "average_loss": average_loss,
        "average_loss_abs": average_loss_abs,
        "profit_loss_ratio": profit_loss_ratio,
        "expectancy": float(clean_series.mean()),
        "max_win": float(clean_series.max()),
        "max_loss": float(clean_series.min()),
        "max_consecutive_losses": int(max_consecutive_losses),
    }


def summarize_group(round_trip_df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarize round-trip pnl by one group column."""

    columns = [group_column] + GROUP_STAT_COLUMNS
    if round_trip_df.empty or group_column not in round_trip_df.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for group_value, group_df in round_trip_df.groupby(group_column, dropna=False):
        pnl_summary = summarize_pnl_series(group_df["net_pnl"])
        gross_pnl = float(pd.to_numeric(group_df["gross_pnl"], errors="coerce").fillna(0.0).sum())
        duration = pd.to_numeric(group_df["duration_minutes"], errors="coerce").dropna()
        rows.append(
            {
                group_column: group_value,
                "trade_count": pnl_summary["count"],
                "gross_pnl": gross_pnl,
                "net_pnl": pnl_summary["total_net_pnl"],
                "win_count": pnl_summary["win_count"],
                "loss_count": pnl_summary["loss_count"],
                "win_rate_pct": pnl_summary["win_rate_pct"],
                "average_pnl": pnl_summary["average_pnl"],
                "median_pnl": pnl_summary["median_pnl"],
                "average_win": pnl_summary["average_win"],
                "average_loss": pnl_summary["average_loss"],
                "profit_loss_ratio": pnl_summary["profit_loss_ratio"],
                "expectancy": pnl_summary["expectancy"],
                "max_win": pnl_summary["max_win"],
                "max_loss": pnl_summary["max_loss"],
                "total_volume": float(pd.to_numeric(group_df["volume"], errors="coerce").fillna(0.0).sum()),
                "avg_duration_minutes": float(duration.mean()) if not duration.empty else None,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def fill_group_stat_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill count/sum fields while keeping rate/average fields nullable."""

    result_df = df.copy()
    for column in ["trade_count", "gross_pnl", "net_pnl", "win_count", "loss_count", "total_volume"]:
        if column in result_df.columns:
            result_df[column] = pd.to_numeric(result_df[column], errors="coerce").fillna(0.0)
    return result_df


def build_side_summary(round_trip_df: pd.DataFrame) -> pd.DataFrame:
    """Build long/short attribution."""

    base_df = pd.DataFrame({"side": ["long", "short"]})
    summary_df = summarize_group(round_trip_df, "side")
    result_df = base_df.merge(summary_df, on="side", how="left")
    return fill_group_stat_nulls(result_df)


def build_hour_summary(round_trip_df: pd.DataFrame) -> pd.DataFrame:
    """Build hour-of-day attribution."""

    base_df = pd.DataFrame({"hour": list(range(24))})
    if not round_trip_df.empty:
        working_df = round_trip_df.copy()
        working_df["hour"] = pd.to_numeric(working_df["entry_hour"], errors="coerce").astype("Int64")
        summary_df = summarize_group(working_df, "hour")
    else:
        summary_df = pd.DataFrame(columns=["hour"] + GROUP_STAT_COLUMNS)
    result_df = base_df.merge(summary_df, on="hour", how="left")
    return fill_group_stat_nulls(result_df)


def build_weekday_summary(round_trip_df: pd.DataFrame) -> pd.DataFrame:
    """Build weekday attribution."""

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    base_df = pd.DataFrame({"weekday": list(range(7)), "weekday_name": weekday_names})
    if not round_trip_df.empty:
        working_df = round_trip_df.copy()
        working_df["weekday"] = pd.to_numeric(working_df["entry_weekday"], errors="coerce").astype("Int64")
        summary_df = summarize_group(working_df, "weekday")
    else:
        summary_df = pd.DataFrame(columns=["weekday"] + GROUP_STAT_COLUMNS)
    result_df = base_df.merge(summary_df, on="weekday", how="left")
    return fill_group_stat_nulls(result_df)


def build_month_summary(daily_df: pd.DataFrame, round_trip_df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly attribution using daily pnl as the exact monthly source."""

    if not daily_df.empty:
        working_daily_df = daily_df.copy()
        working_daily_df["month"] = working_daily_df["date"].dt.to_period("M").astype(str)
        monthly_df = (
            working_daily_df.groupby("month", as_index=False)
            .agg(
                day_count=("date", "count"),
                trade_count=("trade_count", "sum"),
                net_pnl=("net_pnl", "sum"),
                avg_daily_pnl=("net_pnl", "mean"),
                win_day_count=("net_pnl", lambda series: int((series > 0).sum())),
                loss_day_count=("net_pnl", lambda series: int((series < 0).sum())),
                end_balance=("balance", "last"),
            )
            .sort_values("month")
            .reset_index(drop=True)
        )
    elif not round_trip_df.empty:
        monthly_df = (
            round_trip_df.groupby("month", as_index=False)
            .agg(
                day_count=("entry_datetime", lambda series: int(series.dt.date.nunique())),
                trade_count=("net_pnl", "count"),
                net_pnl=("net_pnl", "sum"),
                avg_daily_pnl=("net_pnl", "mean"),
                win_day_count=("net_pnl", lambda series: int((series > 0).sum())),
                loss_day_count=("net_pnl", lambda series: int((series < 0).sum())),
            )
            .sort_values("month")
            .reset_index(drop=True)
        )
        monthly_df["end_balance"] = None
    else:
        return pd.DataFrame(
            columns=[
                "month",
                "day_count",
                "trade_count",
                "round_trip_count",
                "net_pnl",
                "round_trip_net_pnl_estimated",
                "avg_daily_pnl",
                "win_day_count",
                "loss_day_count",
                "end_balance",
            ]
        )

    if not round_trip_df.empty:
        rt_month_df = (
            round_trip_df.groupby("month", as_index=False)
            .agg(
                round_trip_count=("net_pnl", "count"),
                round_trip_net_pnl_estimated=("net_pnl", "sum"),
            )
            .reset_index(drop=True)
        )
        monthly_df = monthly_df.merge(rt_month_df, on="month", how="left")
    else:
        monthly_df["round_trip_count"] = 0
        monthly_df["round_trip_net_pnl_estimated"] = 0.0

    monthly_df["round_trip_count"] = pd.to_numeric(monthly_df["round_trip_count"], errors="coerce").fillna(0).astype(int)
    monthly_df["round_trip_net_pnl_estimated"] = pd.to_numeric(
        monthly_df["round_trip_net_pnl_estimated"], errors="coerce"
    ).fillna(0.0)

    return monthly_df[
        [
            "month",
            "day_count",
            "trade_count",
            "round_trip_count",
            "net_pnl",
            "round_trip_net_pnl_estimated",
            "avg_daily_pnl",
            "win_day_count",
            "loss_day_count",
            "end_balance",
        ]
    ]


def build_daily_extremes(daily_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Build worst and best daily pnl rows."""

    columns = ["bucket", "rank", "date", "net_pnl", "trade_count", "balance", "drawdown", "ddpercent"]
    if daily_df.empty or "net_pnl" not in daily_df.columns:
        return pd.DataFrame(columns=columns)

    sortable_df = daily_df.copy()
    sortable_df["net_pnl"] = pd.to_numeric(sortable_df["net_pnl"], errors="coerce")
    sortable_df = sortable_df.dropna(subset=["net_pnl"])
    if sortable_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[pd.DataFrame] = []
    for bucket, sorted_df in [
        ("worst", sortable_df.sort_values("net_pnl", ascending=True).head(limit)),
        ("best", sortable_df.sort_values("net_pnl", ascending=False).head(limit)),
    ]:
        export_df = sorted_df.copy()
        export_df["bucket"] = bucket
        export_df["rank"] = range(1, len(export_df.index) + 1)
        rows.append(export_df)

    result_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    if "date" in result_df.columns:
        result_df["date"] = result_df["date"].dt.date.astype(str)
    for column in columns:
        if column not in result_df.columns:
            result_df[column] = None
    return result_df[columns]


def trade_count_bucket(value: Any) -> str:
    """Bucket one daily trade count."""

    count = number_or_none(value)
    if count is None or count <= 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    if count <= 40:
        return "21-40"
    if count <= 80:
        return "41-80"
    return "81+"


def build_frequency_distribution(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Build daily trade-frequency distribution."""

    columns = ["trade_count_bucket", "day_count", "net_pnl", "avg_daily_pnl", "median_daily_pnl", "min_daily_pnl", "max_daily_pnl"]
    if daily_df.empty or "trade_count" not in daily_df.columns:
        return pd.DataFrame(columns=columns)

    working_df = daily_df.copy()
    working_df["trade_count_bucket"] = working_df["trade_count"].map(trade_count_bucket)
    bucket_order = ["0", "1-5", "6-10", "11-20", "21-40", "41-80", "81+"]
    summary_df = (
        working_df.groupby("trade_count_bucket", as_index=False)
        .agg(
            day_count=("date", "count"),
            net_pnl=("net_pnl", "sum"),
            avg_daily_pnl=("net_pnl", "mean"),
            median_daily_pnl=("net_pnl", "median"),
            min_daily_pnl=("net_pnl", "min"),
            max_daily_pnl=("net_pnl", "max"),
        )
        .reset_index(drop=True)
    )
    summary_df["bucket_order"] = summary_df["trade_count_bucket"].map({bucket: index for index, bucket in enumerate(bucket_order)})
    summary_df = summary_df.sort_values("bucket_order").drop(columns=["bucket_order"]).reset_index(drop=True)
    return summary_df[columns]


def build_exit_reason_summary(round_trip_df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Build exit reason attribution when metadata exists."""

    columns = ["exit_reason"] + GROUP_STAT_COLUMNS
    if round_trip_df.empty or "exit_reason" not in round_trip_df.columns:
        warnings.append("trades/orders 中没有可用 exit_reason 或 strategy metadata，跳过 exit_reason 分组")
        return pd.DataFrame(columns=columns)

    reason_df = round_trip_df[round_trip_df["exit_reason"].notna() & (round_trip_df["exit_reason"].astype(str).str.len() > 0)]
    if reason_df.empty:
        warnings.append("trades/orders 中没有可用 exit_reason 或 strategy metadata，跳过 exit_reason 分组")
        return pd.DataFrame(columns=columns)
    return summarize_group(reason_df, "exit_reason")


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""

    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False, date_format="iso"))


def detect_peer_report_dir(report_dir: Path) -> Path | None:
    """Detect a sibling cost/no-cost report when directory names follow the project convention."""

    name = report_dir.name
    if "no_cost" in name:
        candidate = report_dir.with_name(name.replace("no_cost", "cost"))
        return candidate if candidate.exists() and candidate != report_dir else None
    if "no-cost" in name:
        candidate = report_dir.with_name(name.replace("no-cost", "cost"))
        return candidate if candidate.exists() and candidate != report_dir else None
    if "nocost" in name:
        candidate = report_dir.with_name(name.replace("nocost", "cost"))
        return candidate if candidate.exists() and candidate != report_dir else None
    if "cost" in name:
        candidate = report_dir.with_name(name.replace("cost", "no_cost", 1))
        return candidate if candidate.exists() and candidate != report_dir else None
    return None


def load_peer_report_summary(peer_dir: Path) -> dict[str, Any] | None:
    """Load enough peer report data for cost-vs-no-cost classification."""

    warnings: list[str] = []
    stats = read_json_optional(peer_dir / "stats.json", warnings)
    diagnostics = read_json_optional(peer_dir / "diagnostics.json", warnings)
    daily_df = read_csv_optional(peer_dir / "daily_pnl.csv", warnings)
    daily_df = prepare_daily_dataframe(daily_df, warnings)
    total_net_pnl = extract_total_net_pnl(stats, diagnostics, daily_df)
    total_commission, total_slippage = extract_total_costs(stats, diagnostics, daily_df)
    return {
        "report_dir": str(peer_dir),
        "total_net_pnl": total_net_pnl,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "total_cost": total_commission + total_slippage,
        "bankrupt": bool(stats.get("bankrupt", diagnostics.get("bankrupt", False))),
        "statistics_valid": bool(stats.get("statistics_valid", False)),
        "warnings": warnings,
    }


def is_no_cost_name(report_dir: Path) -> bool:
    """Return whether the directory name indicates a no-cost report."""

    name = report_dir.name.lower().replace("-", "_")
    return "no_cost" in name or "nocost" in name


def classify_alpha_vs_cost(
    report_dir: Path,
    current_total_net_pnl: float,
    current_total_cost: float,
    current_bankrupt: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """Classify whether losses point to cost drag or negative gross alpha."""

    current_is_no_cost = is_no_cost_name(report_dir) or abs(current_total_cost) <= 1e-9
    peer_dir = detect_peer_report_dir(report_dir)
    peer_summary = load_peer_report_summary(peer_dir) if peer_dir is not None else None

    cost_report: dict[str, Any] | None = None
    no_cost_report: dict[str, Any] | None = None
    current_summary = {
        "report_dir": str(report_dir),
        "total_net_pnl": current_total_net_pnl,
        "total_cost": current_total_cost,
        "bankrupt": current_bankrupt,
    }

    if current_is_no_cost:
        no_cost_report = current_summary
        if peer_summary is not None:
            cost_report = peer_summary
    else:
        cost_report = current_summary
        if peer_summary is not None:
            no_cost_report = peer_summary

    no_cost_pnl = no_cost_report.get("total_net_pnl") if no_cost_report else None
    cost_pnl = cost_report.get("total_net_pnl") if cost_report else None

    gross_alpha_negative = bool(no_cost_pnl <= 0) if no_cost_pnl is not None else None
    cost_drag_dominant = bool(cost_pnl <= 0 and no_cost_pnl > 0) if cost_pnl is not None and no_cost_pnl is not None else None
    candidate_viable_for_oos = bool(cost_pnl > 0 and no_cost_pnl > 0) if cost_pnl is not None and no_cost_pnl is not None else None

    if gross_alpha_negative is True:
        primary_issue = "gross_alpha_negative"
        diagnosis = "毛 alpha 问题：无成本版本 total_net_pnl 仍为负，不能把亏损主要归因于手续费或滑点。"
    elif cost_drag_dominant is True:
        primary_issue = "cost_drag_dominant"
        diagnosis = "成本拖累主导：无成本版本为正，但成本版为负，当前 alpha 不足以覆盖手续费和滑点。"
    elif candidate_viable_for_oos is True:
        primary_issue = "candidate_viable_for_oos"
        diagnosis = "成本版和无成本版均为正，可作为 out-of-sample 候选，但仍不能直接进入 demo。"
    elif no_cost_pnl is not None and no_cost_pnl > 0 and cost_pnl is None:
        primary_issue = "cost_report_missing"
        diagnosis = "无成本版本为正，但缺少成本版对照，无法判断成本后是否仍可存活。"
    else:
        primary_issue = "insufficient_cost_vs_no_cost_evidence"
        diagnosis = "缺少成本版/无成本版成对证据，无法明确判定是成本问题还是毛 alpha 问题。"

    if peer_summary is None:
        warnings.append("未找到同目录命名约定下的成本/无成本 sibling 报告，成本归因只基于当前 REPORT_DIR")
    elif peer_summary.get("warnings"):
        warnings.extend([f"sibling 报告警告: {item}" for item in peer_summary["warnings"]])

    return {
        "current_is_no_cost_report": current_is_no_cost,
        "peer_report_dir": str(peer_dir) if peer_dir is not None else None,
        "cost_report": cost_report,
        "no_cost_report": no_cost_report,
        "cost_report_total_net_pnl": cost_pnl,
        "no_cost_report_total_net_pnl": no_cost_pnl,
        "gross_alpha_negative": gross_alpha_negative,
        "cost_drag_dominant": cost_drag_dominant,
        "candidate_viable_for_oos": candidate_viable_for_oos,
        "primary_issue": primary_issue,
        "diagnosis": diagnosis,
    }


def build_summary(
    artifacts: ReportArtifacts,
    daily_df: pd.DataFrame,
    round_trip_df: pd.DataFrame,
    side_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    weekday_df: pd.DataFrame,
    month_df: pd.DataFrame,
    daily_extremes_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    exit_reason_df: pd.DataFrame,
    timezone_name: str,
    bar_db_check: bool,
) -> dict[str, Any]:
    """Build the attribution summary payload."""

    stats = artifacts.stats
    diagnostics = artifacts.diagnostics
    total_net_pnl = extract_total_net_pnl(stats, diagnostics, daily_df)
    total_commission, total_slippage = extract_total_costs(stats, diagnostics, daily_df)
    total_cost = total_commission + total_slippage
    total_trades = int(
        first_number(
            stats.get("engine_trade_count"),
            stats.get("total_trade_count"),
            diagnostics.get("engine_trade_count"),
            len(artifacts.trades_df.index) if not artifacts.trades_df.empty else None,
        )
        or 0
    )
    inferred_round_trips = int(len(round_trip_df.index))
    round_trip_count = int(first_number(inferred_round_trips if inferred_round_trips > 0 else None, stats.get("closed_round_trip_count")) or 0)

    if not round_trip_df.empty:
        pnl_series = round_trip_df["net_pnl"]
        single_trade_basis = "round_trip"
    else:
        pnl_series = pd.Series([total_net_pnl / total_trades]) if total_trades > 0 else pd.Series(dtype=float)
        single_trade_basis = "engine_trade_fallback"
    pnl_summary = summarize_pnl_series(pnl_series)

    average_single_trade_pnl = pnl_summary["average_pnl"]
    median_single_trade_pnl = pnl_summary["median_pnl"]
    average_daily_trade_count = (
        float(pd.to_numeric(daily_df["trade_count"], errors="coerce").fillna(0.0).mean())
        if not daily_df.empty
        else float(first_number(stats.get("daily_trade_count"), diagnostics.get("daily_trade_count_sum")) or 0.0)
    )
    average_daily_pnl = (
        float(pd.to_numeric(daily_df["net_pnl"], errors="coerce").fillna(0.0).mean())
        if not daily_df.empty
        else float(first_number(stats.get("daily_net_pnl")) or 0.0)
    )

    daily_worst_10 = daily_extremes_df[daily_extremes_df["bucket"] == "worst"].copy() if not daily_extremes_df.empty else pd.DataFrame()
    daily_best_10 = daily_extremes_df[daily_extremes_df["bucket"] == "best"].copy() if not daily_extremes_df.empty else pd.DataFrame()

    alpha_cost_classification = classify_alpha_vs_cost(
        report_dir=artifacts.report_dir,
        current_total_net_pnl=total_net_pnl,
        current_total_cost=total_cost,
        current_bankrupt=bool(stats.get("bankrupt", diagnostics.get("bankrupt", False))),
        warnings=artifacts.warnings,
    )

    round_trip_net = float(pd.to_numeric(round_trip_df["net_pnl"], errors="coerce").fillna(0.0).sum()) if not round_trip_df.empty else 0.0
    round_trip_gross = float(pd.to_numeric(round_trip_df["gross_pnl"], errors="coerce").fillna(0.0).sum()) if not round_trip_df.empty else 0.0

    bar_check_payload = {"requested": bool(bar_db_check), "status": "not_requested"}
    if bar_db_check:
        artifacts.warnings.append("--bar-db-check 已请求，但当前版本不读取数据库，已跳过")
        bar_check_payload = {"requested": True, "status": "skipped", "reason": "database cross-check not implemented"}

    return {
        "report_dir": str(artifacts.report_dir),
        "timezone": timezone_name,
        "warnings": artifacts.warnings,
        "bar_db_check": bar_check_payload,
        "bankrupt": bool(stats.get("bankrupt", diagnostics.get("bankrupt", False))),
        "first_bankrupt_date": diagnostics.get("first_bankrupt_date", stats.get("first_bankrupt_date")),
        "statistics_valid": bool(stats.get("statistics_valid", False)),
        "total_trades": total_trades,
        "round_trip_count": round_trip_count,
        "round_trip_count_inferred_from_trades": inferred_round_trips,
        "single_trade_pnl_basis": single_trade_basis,
        "total_net_pnl": total_net_pnl,
        "round_trip_gross_pnl_estimated": round_trip_gross,
        "round_trip_net_pnl_estimated": round_trip_net,
        "round_trip_vs_report_pnl_diff": round_trip_net - total_net_pnl,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "total_cost": total_cost,
        "average_single_trade_pnl": average_single_trade_pnl,
        "median_single_trade_pnl": median_single_trade_pnl,
        "win_rate_pct": pnl_summary["win_rate_pct"],
        "win_rate": pnl_summary["win_rate"],
        "average_win": pnl_summary["average_win"],
        "average_loss": pnl_summary["average_loss"],
        "average_loss_abs": pnl_summary["average_loss_abs"],
        "profit_loss_ratio": pnl_summary["profit_loss_ratio"],
        "expectancy": pnl_summary["expectancy"],
        "max_single_win": pnl_summary["max_win"],
        "max_single_loss": pnl_summary["max_loss"],
        "max_consecutive_losses": pnl_summary["max_consecutive_losses"],
        "average_daily_trade_count": average_daily_trade_count,
        "average_daily_pnl": average_daily_pnl,
        "worst_10_days": dataframe_records(daily_worst_10),
        "best_10_days": dataframe_records(daily_best_10),
        "by_side": dataframe_records(side_df),
        "by_hour": dataframe_records(hour_df),
        "by_weekday": dataframe_records(weekday_df),
        "by_month": dataframe_records(month_df),
        "trade_frequency_distribution": dataframe_records(frequency_df),
        "exit_reason_summary": dataframe_records(exit_reason_df),
        "alpha_cost_classification": alpha_cost_classification,
        "gross_alpha_negative": alpha_cost_classification["gross_alpha_negative"],
        "cost_drag_dominant": alpha_cost_classification["cost_drag_dominant"],
        "candidate_viable_for_oos": alpha_cost_classification["candidate_viable_for_oos"],
        "primary_issue": alpha_cost_classification["primary_issue"],
        "diagnosis": alpha_cost_classification["diagnosis"],
    }


def format_number(value: Any, digits: int = 6) -> str:
    """Format optional numeric values for markdown."""

    number = number_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a markdown attribution report."""

    warnings = summary.get("warnings") or []
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- 无"

    worst_days = summary.get("worst_10_days") or []
    best_days = summary.get("best_10_days") or []
    worst_lines = "\n".join(
        f"- {row.get('rank')}. {row.get('date')}: net_pnl={format_number(row.get('net_pnl'))}, trades={format_number(row.get('trade_count'), 2)}"
        for row in worst_days[:10]
    ) or "- 无"
    best_lines = "\n".join(
        f"- {row.get('rank')}. {row.get('date')}: net_pnl={format_number(row.get('net_pnl'))}, trades={format_number(row.get('trade_count'), 2)}"
        for row in best_days[:10]
    ) or "- 无"

    frequency_lines = "\n".join(
        f"- {row.get('trade_count_bucket')}: days={row.get('day_count')}, net_pnl={format_number(row.get('net_pnl'))}, avg_daily_pnl={format_number(row.get('avg_daily_pnl'))}"
        for row in (summary.get("trade_frequency_distribution") or [])
    ) or "- 无"

    return (
        "# 交易归因诊断报告\n\n"
        "## 结论\n"
        f"- 主要判断：{summary.get('diagnosis')}\n"
        f"- gross_alpha_negative：{summary.get('gross_alpha_negative')}\n"
        f"- cost_drag_dominant：{summary.get('cost_drag_dominant')}\n"
        f"- candidate_viable_for_oos：{summary.get('candidate_viable_for_oos')}\n"
        f"- primary_issue：{summary.get('primary_issue')}\n\n"
        "## 核心指标\n"
        f"- 总交易次数：{summary.get('total_trades')}\n"
        f"- 往返次数：{summary.get('round_trip_count')}（trades 推断 {summary.get('round_trip_count_inferred_from_trades')}）\n"
        f"- 总净利润：{format_number(summary.get('total_net_pnl'))}\n"
        f"- 平均单笔收益：{format_number(summary.get('average_single_trade_pnl'))}（basis={summary.get('single_trade_pnl_basis')}）\n"
        f"- 中位数单笔收益：{format_number(summary.get('median_single_trade_pnl'))}\n"
        f"- 胜率：{format_number(summary.get('win_rate_pct'))}%\n"
        f"- 平均盈利：{format_number(summary.get('average_win'))}\n"
        f"- 平均亏损：{format_number(summary.get('average_loss'))}\n"
        f"- 盈亏比：{format_number(summary.get('profit_loss_ratio'))}\n"
        f"- 期望值 expectancy：{format_number(summary.get('expectancy'))}\n"
        f"- 最大单笔盈利：{format_number(summary.get('max_single_win'))}\n"
        f"- 最大单笔亏损：{format_number(summary.get('max_single_loss'))}\n"
        f"- 连续亏损次数：{summary.get('max_consecutive_losses')}\n"
        f"- 每日平均交易次数：{format_number(summary.get('average_daily_trade_count'))}\n"
        f"- 每日平均 PnL：{format_number(summary.get('average_daily_pnl'))}\n\n"
        "## 最差 10 天\n"
        f"{worst_lines}\n\n"
        "## 最好 10 天\n"
        f"{best_lines}\n\n"
        "## 交易频率分布\n"
        f"{frequency_lines}\n\n"
        "## 输出文件\n"
        "- attribution_summary.json\n"
        "- attribution_by_side.csv\n"
        "- attribution_by_hour.csv\n"
        "- attribution_by_weekday.csv\n"
        "- attribution_by_month.csv\n"
        "- attribution_daily_worst.csv\n"
        "- attribution_frequency_distribution.csv\n\n"
        "## Warning\n"
        f"{warning_lines}\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Write CSV with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    formats: set[str],
    summary: dict[str, Any],
    side_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    weekday_df: pd.DataFrame,
    month_df: pd.DataFrame,
    daily_extremes_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    exit_reason_df: pd.DataFrame,
    markdown: str,
) -> None:
    """Write selected output formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if "json" in formats:
        write_json(output_dir / "attribution_summary.json", summary)

    if "csv" in formats:
        write_dataframe(output_dir / "attribution_by_side.csv", side_df)
        write_dataframe(output_dir / "attribution_by_hour.csv", hour_df)
        write_dataframe(output_dir / "attribution_by_weekday.csv", weekday_df)
        write_dataframe(output_dir / "attribution_by_month.csv", month_df)
        write_dataframe(output_dir / "attribution_daily_worst.csv", daily_extremes_df)
        write_dataframe(output_dir / "attribution_frequency_distribution.csv", frequency_df)
        if not exit_reason_df.empty:
            write_dataframe(output_dir / "attribution_by_exit_reason.csv", exit_reason_df)

    if "md" in formats:
        (output_dir / "attribution_report.md").write_text(markdown, encoding="utf-8")


def run_analysis(
    report_dir: Path,
    output_dir: Path,
    timezone_name: str,
    formats: set[str],
    bar_db_check: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run the full trade attribution workflow."""

    ZoneInfo(timezone_name)
    artifacts = load_report_artifacts(report_dir)
    if not artifacts.stats and artifacts.trades_df.empty and artifacts.daily_df.empty:
        artifacts.warnings.append("stats/trades/daily 都不可用，报告信息严重不足")

    daily_df = prepare_daily_dataframe(artifacts.daily_df, artifacts.warnings)
    trades_df = prepare_trades_dataframe(artifacts.trades_df, timezone_name, artifacts.warnings)
    round_trip_df = pair_round_trips(trades_df, artifacts.orders_df, artifacts.warnings)
    total_commission, total_slippage = extract_total_costs(artifacts.stats, artifacts.diagnostics, daily_df)
    round_trip_df = apply_pnl_scaling(round_trip_df, artifacts.stats, total_commission, total_slippage, artifacts.warnings)

    side_df = build_side_summary(round_trip_df)
    hour_df = build_hour_summary(round_trip_df)
    weekday_df = build_weekday_summary(round_trip_df)
    month_df = build_month_summary(daily_df, round_trip_df)
    daily_extremes_df = build_daily_extremes(daily_df, limit=10)
    frequency_df = build_frequency_distribution(daily_df)
    exit_reason_df = build_exit_reason_summary(round_trip_df, artifacts.warnings)

    summary = build_summary(
        artifacts=artifacts,
        daily_df=daily_df,
        round_trip_df=round_trip_df,
        side_df=side_df,
        hour_df=hour_df,
        weekday_df=weekday_df,
        month_df=month_df,
        daily_extremes_df=daily_extremes_df,
        frequency_df=frequency_df,
        exit_reason_df=exit_reason_df,
        timezone_name=timezone_name,
        bar_db_check=bar_db_check,
    )
    markdown = render_markdown(summary)
    write_outputs(
        output_dir=output_dir,
        formats=formats,
        summary=summary,
        side_df=side_df,
        hour_df=hour_df,
        weekday_df=weekday_df,
        month_df=month_df,
        daily_extremes_df=daily_extremes_df,
        frequency_df=frequency_df,
        exit_reason_df=exit_reason_df,
        markdown=markdown,
    )

    log_event(
        logger,
        logging.INFO,
        "trade_attribution.completed",
        "Trade attribution analysis completed",
        report_dir=report_dir,
        output_dir=output_dir,
        total_net_pnl=summary.get("total_net_pnl"),
        gross_alpha_negative=summary.get("gross_alpha_negative"),
        cost_drag_dominant=summary.get("cost_drag_dominant"),
    )
    return summary


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("analyze_trade_attribution", verbose=args.verbose)

    try:
        report_dir = resolve_path(args.report_dir)
        output_dir = resolve_path(args.output_dir, default_path=report_dir / "trade_attribution")
        formats = normalize_formats(args.format)
        summary = run_analysis(
            report_dir=report_dir,
            output_dir=output_dir,
            timezone_name=args.timezone,
            formats=formats,
            bar_db_check=args.bar_db_check,
            logger=logger,
        )

        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        else:
            print_json_block(
                "Trade attribution summary:",
                {
                    "report_dir": summary.get("report_dir"),
                    "output_dir": str(output_dir),
                    "total_net_pnl": summary.get("total_net_pnl"),
                    "round_trip_count": summary.get("round_trip_count"),
                    "gross_alpha_negative": summary.get("gross_alpha_negative"),
                    "cost_drag_dominant": summary.get("cost_drag_dominant"),
                    "candidate_viable_for_oos": summary.get("candidate_viable_for_oos"),
                    "warnings": summary.get("warnings"),
                },
            )
        return 0
    except TradeAttributionError as exc:
        log_event(logger, logging.ERROR, "trade_attribution.error", str(exc))
        return 1
    except Exception:
        logger.exception(
            "Unexpected error during trade attribution analysis",
            extra={"event": "trade_attribution.unexpected_error"},
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
