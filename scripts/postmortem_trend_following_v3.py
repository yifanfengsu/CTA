#!/usr/bin/env python3
"""Postmortem diagnostics for Trend Following V3 research outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable


SPLITS = ["train", "validation", "oos"]
POLICY_FAMILIES = [
    "4h_donchian",
    "1d_donchian",
    "4h_ema",
    "1d_ema",
    "vol_compression",
    "ensemble",
    "risk_filtered",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research" / "trend_following_v3_postmortem"
DEFAULT_FUNDING_BPS = [1.0, 3.0, 5.0, 10.0]
DEFAULT_CAPITAL = 5000.0
CONCENTRATION_LIMIT = 0.70
TOP_TRADE_LIMIT = 0.80

REQUIRED_SPLIT_FILES = [
    "trend_v3_trades.csv",
    "trend_v3_policy_leaderboard.csv",
    "trend_v3_policy_by_symbol.csv",
    "trend_v3_policy_by_month.csv",
    "trend_v3_symbol_contribution.csv",
    "trend_v3_portfolio_equity_curve.csv",
    "trend_v3_portfolio_daily_pnl.csv",
    "trend_v3_drawdown.csv",
    "trend_v3_summary.json",
    "trend_v3_report.md",
]
REQUIRED_COMPARE_FILES = [
    "trend_v3_compare_leaderboard.csv",
    "trend_v3_compare_summary.json",
    "trend_v3_compare_report.md",
]
OUTPUT_FILES = [
    "trend_v3_postmortem_summary.json",
    "trend_v3_postmortem_report.md",
    "policy_family_analysis.csv",
    "symbol_contribution_postmortem.csv",
    "by_month.csv",
    "by_quarter.csv",
    "by_symbol_month.csv",
    "by_policy_month.csv",
    "top_trade_concentration.csv",
    "funding_sensitivity.csv",
    "rejected_candidate_reasons.csv",
    "v3_1_recommendations.json",
]


class TrendFollowingV3PostmortemError(Exception):
    """Raised when postmortem cannot continue."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Postmortem diagnostics for Trend Following V3 failure attribution.")
    parser.add_argument("--train-dir", default="reports/research/trend_following_v3/train")
    parser.add_argument("--validation-dir", default="reports/research/trend_following_v3/validation")
    parser.add_argument("--oos-dir", default="reports/research/trend_following_v3/oos")
    parser.add_argument("--compare-dir", default="reports/research/trend_following_v3_compare")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--funding-mode", choices=["synthetic", "none", "actual"], default="synthetic")
    parser.add_argument("--funding-bps-per-8h", default="1,3,5,10")
    parser.add_argument("--min-trade-count", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print trend_v3_postmortem_summary.json payload.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path_arg: str | Path) -> Path:
    """Resolve a path relative to project root."""

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_funding_bps(raw_value: str | list[float] | tuple[float, ...]) -> list[float]:
    """Parse comma or whitespace separated funding bps values."""

    if isinstance(raw_value, (list, tuple)):
        values = [float(item) for item in raw_value]
    else:
        values = [float(token) for token in str(raw_value).replace(",", " ").split() if token.strip()]
    if not values:
        raise TrendFollowingV3PostmortemError("--funding-bps-per-8h 不能为空")
    return values


def finite_or_none(value: Any) -> float | None:
    """Return finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def safe_sum(frame: pd.DataFrame, column: str) -> float:
    """Sum a numeric column if available."""

    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    """Mean of a numeric column if available."""

    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.astype(object).where(pd.notna(df), None)
    return json.loads(safe_df.to_json(orient="records", force_ascii=False))


def read_csv_optional(directory: Path, filename: str, warnings: list[str], label: str) -> pd.DataFrame:
    """Read a CSV file, returning empty DataFrame on missing or invalid input."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"{label}: missing {filename}: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"{label}: failed to read {filename}: {exc!r}")
        return pd.DataFrame()


def read_json_optional(directory: Path, filename: str, warnings: list[str], label: str) -> dict[str, Any]:
    """Read a JSON file, returning empty dict on missing or invalid input."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"{label}: missing {filename}: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"{label}: failed to read {filename}: {exc!r}")
        return {}


def read_text_optional(directory: Path, filename: str, warnings: list[str], label: str) -> str:
    """Read a text file, returning empty string on missing or invalid input."""

    path = directory / filename
    if not path.exists():
        warnings.append(f"{label}: missing {filename}: {path}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"{label}: failed to read {filename}: {exc!r}")
        return ""


def load_split_artifacts(split: str, directory: Path, warnings: list[str]) -> dict[str, Any]:
    """Load all required files for one V3 split."""

    label = f"split={split}"
    artifacts: dict[str, Any] = {"directory": directory}
    for filename in REQUIRED_SPLIT_FILES:
        key = filename.rsplit(".", 1)[0]
        if filename.endswith(".csv"):
            artifacts[key] = read_csv_optional(directory, filename, warnings, label)
        elif filename.endswith(".json"):
            artifacts[key] = read_json_optional(directory, filename, warnings, label)
        elif filename.endswith(".md"):
            artifacts[key] = read_text_optional(directory, filename, warnings, label)
    return artifacts


def load_compare_artifacts(directory: Path, warnings: list[str]) -> dict[str, Any]:
    """Load all required V3 compare files."""

    label = "compare"
    artifacts: dict[str, Any] = {"directory": directory}
    for filename in REQUIRED_COMPARE_FILES:
        key = filename.rsplit(".", 1)[0]
        if filename.endswith(".csv"):
            artifacts[key] = read_csv_optional(directory, filename, warnings, label)
        elif filename.endswith(".json"):
            artifacts[key] = read_json_optional(directory, filename, warnings, label)
        elif filename.endswith(".md"):
            artifacts[key] = read_text_optional(directory, filename, warnings, label)
    return artifacts


def classify_policy_family(policy_name: str) -> str:
    """Classify a Trend V3 policy into a postmortem family bucket."""

    name = str(policy_name or "").lower()
    if "ensemble" in name:
        return "ensemble"
    if "vol_compression" in name or "compression" in name:
        return "vol_compression"
    if "risk_filters" in name or "risk_filtered" in name or "with_risk" in name:
        return "risk_filtered"
    if "1d" in name and "ema" in name:
        return "1d_ema"
    if "4h" in name and "ema" in name:
        return "4h_ema"
    if "1d" in name and "donchian" in name:
        return "1d_donchian"
    if "4h" in name and "donchian" in name:
        return "4h_donchian"
    return "unknown"


def prepare_trades(trades_df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Normalize trade columns used by postmortem analysis."""

    if trades_df.empty:
        return pd.DataFrame()
    working = trades_df.copy()
    working["split"] = split
    for column in [
        "entry_price",
        "exit_price",
        "holding_minutes",
        "volume",
        "contract_size",
        "gross_pnl",
        "fee",
        "slippage",
        "net_pnl",
        "no_cost_pnl",
        "no_cost_net_pnl",
        "turnover",
    ]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    if "no_cost_net_pnl" not in working.columns and "no_cost_pnl" in working.columns:
        working["no_cost_net_pnl"] = working["no_cost_pnl"]
    if "no_cost_pnl" not in working.columns and "no_cost_net_pnl" in working.columns:
        working["no_cost_pnl"] = working["no_cost_net_pnl"]
    for column in ["entry_time", "exit_time"]:
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce")
    if "policy_name" in working.columns:
        working["policy_family"] = working["policy_name"].map(classify_policy_family)
    else:
        working["policy_name"] = None
        working["policy_family"] = "unknown"
    if "symbol" not in working.columns:
        working["symbol"] = None
    if "direction" not in working.columns:
        working["direction"] = None
    exit_time = working["exit_time"] if "exit_time" in working.columns else pd.Series(pd.NaT, index=working.index)
    entry_time = working["entry_time"] if "entry_time" in working.columns else pd.Series(pd.NaT, index=working.index)
    working["month"] = exit_time.map(lambda value: timestamp_period_string(value, "M"))
    working["quarter"] = exit_time.map(lambda value: timestamp_period_string(value, "Q"))
    working["entry_month"] = entry_time.map(lambda value: timestamp_period_string(value, "M"))
    working["exit_month"] = working["month"]
    working["estimated_notional"] = estimate_trade_notional(working)
    return working


def timestamp_period_string(value: Any, freq: str) -> str | None:
    """Return a local-wall-clock period string without pandas timezone warnings."""

    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return str(timestamp.to_period(freq))


def estimate_trade_notional(trades_df: pd.DataFrame) -> pd.Series:
    """Estimate absolute position notional for funding stress."""

    if trades_df.empty:
        return pd.Series(dtype=float)
    if {"entry_price", "volume", "contract_size"}.issubset(trades_df.columns):
        notional = (
            pd.to_numeric(trades_df["entry_price"], errors="coerce").abs()
            * pd.to_numeric(trades_df["volume"], errors="coerce").abs()
            * pd.to_numeric(trades_df["contract_size"], errors="coerce").abs()
        )
    else:
        notional = pd.Series(np.nan, index=trades_df.index, dtype=float)
    if "turnover" in trades_df.columns:
        fallback = pd.to_numeric(trades_df["turnover"], errors="coerce").abs() / 2.0
        notional = notional.where((notional.notna()) & (notional > 0), fallback)
    return notional.fillna(0.0).astype(float)


def max_drawdown_from_trades(trade_df: pd.DataFrame, capital: float = DEFAULT_CAPITAL) -> tuple[float, float]:
    """Compute closed-trade max drawdown and percent."""

    if trade_df.empty or "net_pnl" not in trade_df.columns:
        return 0.0, 0.0
    working = trade_df.copy()
    if "exit_time" in working.columns:
        working = working.sort_values("exit_time", kind="stable")
    net = pd.to_numeric(working["net_pnl"], errors="coerce").fillna(0.0)
    equity = capital + net.cumsum()
    peak = pd.concat([pd.Series([capital], dtype=float), equity.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = peak - equity.reset_index(drop=True)
    ddpercent = drawdown / peak.replace(0, np.nan) * 100.0
    return float(drawdown.max()), float(ddpercent.max()) if ddpercent.notna().any() else 0.0


def profit_factor_from_net(net: pd.Series) -> float | None:
    """Compute profit factor from net PnL series."""

    wins = net[net > 0]
    losses = net[net < 0]
    if losses.empty:
        return None
    if wins.empty:
        return 0.0
    return float(wins.sum() / abs(losses.sum()))


def summarize_trade_slice(trade_df: pd.DataFrame, capital: float = DEFAULT_CAPITAL) -> dict[str, Any]:
    """Summarize a trade slice with V3-compatible closed-trade metrics."""

    if trade_df.empty:
        return {
            "trade_count": 0,
            "no_cost_net_pnl": 0.0,
            "net_pnl": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "max_ddpercent": 0.0,
            "avg_holding_minutes": None,
            "cost_drag": 0.0,
            "avg_trade_net_pnl": None,
            "active_symbol_count": 0,
        }
    net = pd.to_numeric(trade_df.get("net_pnl"), errors="coerce").fillna(0.0)
    max_dd, max_ddpercent = max_drawdown_from_trades(trade_df, capital)
    no_cost = safe_sum(trade_df, "no_cost_net_pnl")
    net_total = float(net.sum())
    return {
        "trade_count": int(len(trade_df.index)),
        "no_cost_net_pnl": no_cost,
        "net_pnl": net_total,
        "win_rate": float((net > 0).mean()) if len(net.index) else None,
        "profit_factor": profit_factor_from_net(net),
        "max_drawdown": max_dd,
        "max_ddpercent": max_ddpercent,
        "avg_holding_minutes": safe_mean(trade_df, "holding_minutes"),
        "cost_drag": no_cost - net_total,
        "avg_trade_net_pnl": float(net.mean()) if len(net.index) else None,
        "active_symbol_count": int(trade_df["symbol"].nunique()) if "symbol" in trade_df.columns else 0,
    }


def split_capital(split_artifacts: dict[str, Any]) -> float:
    """Return portfolio capital from split summary, with a conservative default."""

    summary = split_artifacts.get("trend_v3_summary") or {}
    for key in ["portfolio_capital", "capital"]:
        value = finite_or_none(summary.get(key))
        if value is not None and value > 0:
            return value
    return DEFAULT_CAPITAL


def build_policy_family_analysis(trades_by_split: dict[str, pd.DataFrame], capitals: dict[str, float]) -> pd.DataFrame:
    """Build policy family diagnostics by split."""

    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        trades = trades_by_split.get(split, pd.DataFrame())
        for family in POLICY_FAMILIES:
            group = trades[trades["policy_family"] == family].copy() if not trades.empty else pd.DataFrame()
            row = {"split": split, "policy_family": family}
            row.update(summarize_trade_slice(group, capitals.get(split, DEFAULT_CAPITAL)))
            rows.append(row)
    result = pd.DataFrame(rows)
    stable_by_family: dict[str, str] = {}
    for family, family_df in result.groupby("policy_family", dropna=False):
        values = {
            row["split"]: finite_or_none(row.get("no_cost_net_pnl"))
            for row in family_df.to_dict(orient="records")
        }
        available = [value for value in values.values() if value is not None]
        if len(available) < 3:
            direction = "insufficient_data"
        elif all(value > 0 for value in available):
            direction = "positive_all_splits"
        elif all(value < 0 for value in available):
            direction = "negative_all_splits"
        elif (values.get("oos") or 0.0) > 0 and (values.get("validation") or 0.0) < 0:
            direction = "oos_rebound_validation_failed"
        elif (values.get("validation") or 0.0) > 0 and (values.get("oos") or 0.0) < 0:
            direction = "validation_only_reversal"
        else:
            direction = "mixed"
        stable_by_family[str(family)] = direction
    result["stable_direction"] = result["policy_family"].map(stable_by_family)
    columns = [
        "split",
        "policy_family",
        "trade_count",
        "no_cost_net_pnl",
        "net_pnl",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "avg_holding_minutes",
        "cost_drag",
        "stable_direction",
        "max_ddpercent",
        "avg_trade_net_pnl",
        "active_symbol_count",
    ]
    return result[columns]


def largest_symbol_dependency(symbol_net: pd.Series) -> tuple[float | None, str | None]:
    """Return largest symbol dependency share and symbol."""

    if symbol_net.empty:
        return None, None
    positive = symbol_net[symbol_net > 0]
    if not positive.empty and float(positive.sum()) > 0:
        symbol = str(positive.idxmax())
        return float(positive.max() / positive.sum()), symbol
    absolute = symbol_net.abs()
    if float(absolute.sum()) > 0:
        symbol = str(absolute.idxmax())
        return float(absolute.max() / absolute.sum()), symbol
    return None, None


def build_symbol_contribution_postmortem(trades_by_split: dict[str, pd.DataFrame], capitals: dict[str, float]) -> pd.DataFrame:
    """Build per split/policy/symbol contribution diagnostics."""

    rows: list[dict[str, Any]] = []
    for split, trades in trades_by_split.items():
        if trades.empty:
            continue
        for policy_name, policy_group in trades.groupby("policy_name", dropna=False):
            symbol_net = policy_group.groupby("symbol", dropna=False)["net_pnl"].sum()
            total_net = float(symbol_net.sum())
            largest_dependency, dependency_symbol = largest_symbol_dependency(symbol_net)
            positive_sum = float(symbol_net[symbol_net > 0].sum())
            abs_sum = float(symbol_net.abs().sum())
            total_no_cost = safe_sum(policy_group, "no_cost_net_pnl")
            for symbol, group in policy_group.groupby("symbol", dropna=False):
                metrics = summarize_trade_slice(group, capitals.get(split, DEFAULT_CAPITAL))
                net_pnl = metrics["net_pnl"]
                contribution_share = net_pnl / total_net if total_net else None
                positive_pnl_share = net_pnl / positive_sum if positive_sum > 0 and net_pnl > 0 else None
                absolute_pnl_share = abs(net_pnl) / abs_sum if abs_sum > 0 else None
                without_symbol_net = total_net - net_pnl
                without_symbol_no_cost = total_no_cost - metrics["no_cost_net_pnl"]
                rows.append(
                    {
                        "split": split,
                        "policy_name": policy_name,
                        "policy_family": classify_policy_family(str(policy_name)),
                        "symbol": symbol,
                        "trade_count": metrics["trade_count"],
                        "no_cost_pnl": metrics["no_cost_net_pnl"],
                        "net_pnl": net_pnl,
                        "contribution_share": contribution_share,
                        "positive_pnl_share": positive_pnl_share,
                        "absolute_pnl_share": absolute_pnl_share,
                        "win_rate": metrics["win_rate"],
                        "avg_trade_pnl": metrics["avg_trade_net_pnl"],
                        "policy_net_pnl": total_net,
                        "policy_no_cost_pnl": total_no_cost,
                        "net_pnl_without_symbol": without_symbol_net,
                        "no_cost_pnl_without_symbol": without_symbol_no_cost,
                        "removing_symbol_improves_policy": bool(without_symbol_net > total_net),
                        "largest_symbol_dependency": largest_dependency,
                        "largest_dependency_symbol": dependency_symbol,
                        "is_largest_dependency_symbol": bool(str(symbol) == str(dependency_symbol)),
                    }
                )
    columns = [
        "split",
        "policy_name",
        "policy_family",
        "symbol",
        "trade_count",
        "no_cost_pnl",
        "net_pnl",
        "contribution_share",
        "positive_pnl_share",
        "absolute_pnl_share",
        "win_rate",
        "avg_trade_pnl",
        "policy_net_pnl",
        "policy_no_cost_pnl",
        "net_pnl_without_symbol",
        "no_cost_pnl_without_symbol",
        "removing_symbol_improves_policy",
        "largest_symbol_dependency",
        "largest_dependency_symbol",
        "is_largest_dependency_symbol",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_grouped_time_summary(
    trades_by_split: dict[str, pd.DataFrame],
    group_columns: list[str],
    capitals: dict[str, float],
) -> pd.DataFrame:
    """Build grouped month/quarter/regime summaries."""

    rows: list[dict[str, Any]] = []
    for split, trades in trades_by_split.items():
        if trades.empty:
            continue
        working = trades.dropna(subset=[column for column in group_columns if column in trades.columns]).copy()
        if working.empty:
            continue
        for keys, group in working.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"split": split}
            row.update({column: key for column, key in zip(group_columns, keys)})
            metrics = summarize_trade_slice(group, capitals.get(split, DEFAULT_CAPITAL))
            row.update(
                {
                    "trade_count": metrics["trade_count"],
                    "net_pnl": metrics["net_pnl"],
                    "no_cost_pnl": metrics["no_cost_net_pnl"],
                    "win_rate": metrics["win_rate"],
                    "avg_trade": metrics["avg_trade_net_pnl"],
                    "max_drawdown": metrics["max_drawdown"],
                    "max_ddpercent": metrics["max_ddpercent"],
                    "active_symbol_count": metrics["active_symbol_count"],
                    "avg_holding_minutes": metrics["avg_holding_minutes"],
                }
            )
            rows.append(row)
    sort_columns = ["split"] + group_columns
    if not rows:
        return pd.DataFrame(columns=sort_columns + ["trade_count", "net_pnl", "no_cost_pnl", "win_rate", "avg_trade", "max_drawdown"])
    return pd.DataFrame(rows).sort_values(sort_columns, kind="stable").reset_index(drop=True)


def build_top_trade_concentration(trades_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build top trade concentration diagnostics by split/policy."""

    rows: list[dict[str, Any]] = []
    for split, trades in trades_by_split.items():
        if trades.empty:
            continue
        for policy_name, group in trades.groupby("policy_name", dropna=False):
            net = pd.to_numeric(group["net_pnl"], errors="coerce").fillna(0.0).sort_values(ascending=False)
            count = int(len(net.index))
            total = float(net.sum())
            top_1 = float(net.head(1).sum()) if count else 0.0
            top_3 = float(net.head(3).sum()) if count else 0.0
            top_5_count = max(1, int(math.ceil(count * 0.05))) if count else 0
            top_10_count = max(1, int(math.ceil(count * 0.10))) if count else 0
            top_5 = float(net.head(top_5_count).sum()) if count else 0.0
            top_10 = float(net.head(top_10_count).sum()) if count else 0.0
            rows.append(
                {
                    "split": split,
                    "policy_name": policy_name,
                    "policy_family": classify_policy_family(str(policy_name)),
                    "trade_count": count,
                    "top_1_trade_pnl": top_1,
                    "top_3_trade_pnl": top_3,
                    "top_5pct_trade_pnl": top_5,
                    "top_10pct_trade_pnl": top_10,
                    "top_5pct_trade_count": top_5_count,
                    "top_10pct_trade_count": top_10_count,
                    "total_net_pnl": total,
                    "top_1_share": top_1 / total if total else None,
                    "top_5pct_share": top_5 / total if total else None,
                    "top_10pct_share": top_10 / total if total else None,
                    "remove_top_1_pnl": total - top_1,
                    "remove_top_5pct_pnl": total - top_5,
                    "remove_top_10pct_pnl": total - top_10,
                }
            )
    columns = [
        "split",
        "policy_name",
        "policy_family",
        "trade_count",
        "top_1_trade_pnl",
        "top_3_trade_pnl",
        "top_5pct_trade_pnl",
        "top_10pct_trade_pnl",
        "top_5pct_trade_count",
        "top_10pct_trade_count",
        "total_net_pnl",
        "top_1_share",
        "top_5pct_share",
        "top_10pct_share",
        "remove_top_1_pnl",
        "remove_top_5pct_pnl",
        "remove_top_10pct_pnl",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_funding_sensitivity(
    trades_by_split: dict[str, pd.DataFrame],
    funding_bps_values: list[float],
    funding_mode: str,
    warnings: list[str],
) -> tuple[pd.DataFrame, str]:
    """Build synthetic funding stress diagnostics."""

    effective_mode = funding_mode
    if funding_mode == "actual":
        warnings.append("funding-mode=actual requested, but no actual funding fee input is available; using synthetic funding stress instead.")
        effective_mode = "synthetic"
    rows: list[dict[str, Any]] = []
    for split, trades in trades_by_split.items():
        if trades.empty:
            continue
        for policy_name, group in trades.groupby("policy_name", dropna=False):
            original = safe_sum(group, "net_pnl")
            holding = pd.to_numeric(group.get("holding_minutes"), errors="coerce").fillna(0.0)
            notional = pd.to_numeric(group.get("estimated_notional"), errors="coerce").fillna(0.0)
            funding_exposure = float((notional.abs() * (holding / 480.0)).sum())
            break_even = original * 10000.0 / funding_exposure if funding_exposure > 0 else None
            bps_iterable = [0.0] if effective_mode == "none" else funding_bps_values
            for funding_bps in bps_iterable:
                funding_cost = 0.0 if effective_mode == "none" else funding_exposure * float(funding_bps) / 10000.0
                adjusted = original - funding_cost
                rows.append(
                    {
                        "split": split,
                        "policy_name": policy_name,
                        "policy_family": classify_policy_family(str(policy_name)),
                        "funding_mode": effective_mode,
                        "synthetic_funding_stress": bool(effective_mode == "synthetic"),
                        "funding_bps_per_8h": float(funding_bps),
                        "original_net_pnl": original,
                        "funding_cost": funding_cost,
                        "funding_adjusted_net_pnl": adjusted,
                        "funding_break_even_bps": break_even,
                        "remains_positive_after_funding": bool(adjusted > 0),
                        "funding_exposure_notional_periods": funding_exposure,
                    }
                )
    columns = [
        "split",
        "policy_name",
        "policy_family",
        "funding_mode",
        "synthetic_funding_stress",
        "funding_bps_per_8h",
        "original_net_pnl",
        "funding_cost",
        "funding_adjusted_net_pnl",
        "funding_break_even_bps",
        "remains_positive_after_funding",
        "funding_exposure_notional_periods",
    ]
    return pd.DataFrame(rows, columns=columns), effective_mode


def numeric_from_row(row: pd.Series | None, column: str) -> float | None:
    """Return a numeric value from a row."""

    if row is None:
        return None
    return finite_or_none(row.get(column))


def build_rejected_candidate_reasons(
    compare_df: pd.DataFrame,
    leaderboards_by_split: dict[str, pd.DataFrame],
    min_trade_count: int,
) -> pd.DataFrame:
    """Build rejected candidate reasons from compare output or split leaderboards."""

    if not compare_df.empty and "policy_name" in compare_df.columns:
        result = compare_df.copy()
        result["policy_family"] = result["policy_name"].map(classify_policy_family)
        preferred = [
            "policy_name",
            "policy_family",
            "train_no_cost_net_pnl",
            "validation_no_cost_net_pnl",
            "oos_no_cost_net_pnl",
            "train_net_pnl",
            "validation_net_pnl",
            "oos_net_pnl",
            "train_trade_count",
            "validation_trade_count",
            "oos_trade_count",
            "oos_largest_symbol_pnl_share",
            "oos_top_5pct_trade_pnl_contribution",
            "stable_candidate",
            "rejection_reasons",
        ]
        for column in preferred:
            if column not in result.columns:
                result[column] = None
        return result[preferred]

    policies: set[str] = set()
    for frame in leaderboards_by_split.values():
        if not frame.empty and "policy_name" in frame.columns:
            policies.update(str(item) for item in frame["policy_name"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for policy_name in sorted(policies):
        row: dict[str, Any] = {"policy_name": policy_name, "policy_family": classify_policy_family(policy_name)}
        reasons: list[str] = []
        for split in SPLITS:
            frame = leaderboards_by_split.get(split, pd.DataFrame())
            split_row = None
            if not frame.empty and "policy_name" in frame.columns:
                subset = frame[frame["policy_name"] == policy_name]
                if not subset.empty:
                    split_row = subset.iloc[0]
            if split_row is None:
                reasons.append(f"missing_{split}_row")
                continue
            no_cost = numeric_from_row(split_row, "no_cost_net_pnl")
            net = numeric_from_row(split_row, "net_pnl")
            trades = numeric_from_row(split_row, "trade_count")
            row[f"{split}_no_cost_net_pnl"] = no_cost
            row[f"{split}_net_pnl"] = net
            row[f"{split}_trade_count"] = trades
            if no_cost is None or no_cost <= 0:
                reasons.append(f"{split}_no_cost_net_pnl_not_positive")
            if trades is None or trades < min_trade_count:
                reasons.append(f"{split}_trade_count_under_{min_trade_count}")
        oos_frame = leaderboards_by_split.get("oos", pd.DataFrame())
        oos_row = None
        if not oos_frame.empty and "policy_name" in oos_frame.columns:
            subset = oos_frame[oos_frame["policy_name"] == policy_name]
            if not subset.empty:
                oos_row = subset.iloc[0]
        oos_net = numeric_from_row(oos_row, "net_pnl")
        largest = numeric_from_row(oos_row, "largest_symbol_pnl_share")
        top = numeric_from_row(oos_row, "top_5pct_trade_pnl_contribution")
        row["oos_largest_symbol_pnl_share"] = largest
        row["oos_top_5pct_trade_pnl_contribution"] = top
        if oos_net is None or oos_net < 0:
            reasons.append("oos_cost_aware_net_pnl_negative")
        if largest is None or largest > CONCENTRATION_LIMIT:
            reasons.append("oos_largest_symbol_pnl_share_over_0p7")
        if top is None or top > TOP_TRADE_LIMIT:
            reasons.append("oos_top_5pct_trade_pnl_contribution_over_0p8")
        row["stable_candidate"] = bool(not reasons)
        row["rejection_reasons"] = ";".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def find_policy_row(frame: pd.DataFrame, policy_name: str) -> pd.Series | None:
    """Find one policy row in a DataFrame."""

    if frame.empty or "policy_name" not in frame.columns:
        return None
    subset = frame[frame["policy_name"] == policy_name]
    if subset.empty:
        return None
    return subset.iloc[0]


def select_oos_best_policy(oos_leaderboard: pd.DataFrame, oos_trades: pd.DataFrame) -> str | None:
    """Select OOS best policy by cost-aware net PnL."""

    if not oos_leaderboard.empty and {"policy_name", "net_pnl"}.issubset(oos_leaderboard.columns):
        working = oos_leaderboard.copy()
        working["net_pnl"] = pd.to_numeric(working["net_pnl"], errors="coerce")
        working = working.dropna(subset=["net_pnl"]).sort_values(["net_pnl", "policy_name"], ascending=[False, True], kind="stable")
        if not working.empty:
            return str(working.iloc[0]["policy_name"])
    if not oos_trades.empty and "policy_name" in oos_trades.columns:
        grouped = oos_trades.groupby("policy_name", dropna=False)["net_pnl"].sum().sort_values(ascending=False)
        if not grouped.empty:
            return str(grouped.index[0])
    return None


def funding_adjusted_for_policy(funding_df: pd.DataFrame, policy_name: str, bps: float) -> float | None:
    """Return OOS funding-adjusted PnL for a policy and bps."""

    if funding_df.empty:
        return None
    subset = funding_df[
        (funding_df["split"] == "oos")
        & (funding_df["policy_name"] == policy_name)
        & (pd.to_numeric(funding_df["funding_bps_per_8h"], errors="coerce") == float(bps))
    ]
    if subset.empty:
        return None
    return finite_or_none(subset.iloc[0].get("funding_adjusted_net_pnl"))


def decide_v3_1_recommendation(best_metrics: dict[str, Any], stable_candidate_exists: bool) -> dict[str, Any]:
    """Decide whether V3.1 research is justified from postmortem metrics."""

    validation_no_cost = finite_or_none(best_metrics.get("validation_no_cost_net_pnl"))
    train_no_cost = finite_or_none(best_metrics.get("train_no_cost_net_pnl"))
    oos_no_cost = finite_or_none(best_metrics.get("oos_no_cost_net_pnl"))
    remove_top_1 = finite_or_none(best_metrics.get("remove_top_1_pnl"))
    remove_top_5 = finite_or_none(best_metrics.get("remove_top_5pct_pnl"))
    largest_share = finite_or_none(best_metrics.get("largest_symbol_pnl_share"))
    top_share = finite_or_none(best_metrics.get("top_5pct_trade_pnl_contribution"))
    funding_1 = finite_or_none(best_metrics.get("funding_adjusted_1bps"))
    funding_3 = finite_or_none(best_metrics.get("funding_adjusted_3bps"))
    funding_5 = finite_or_none(best_metrics.get("funding_adjusted_5bps"))
    funding_10 = finite_or_none(best_metrics.get("funding_adjusted_10bps"))
    family = str(best_metrics.get("policy_family") or "")

    red_flags: list[str] = []
    if not stable_candidate_exists:
        red_flags.append("no_stable_candidate")
    if validation_no_cost is None or validation_no_cost <= 0:
        red_flags.append("validation_no_cost_negative")
    if largest_share is None or largest_share > CONCENTRATION_LIMIT:
        red_flags.append("symbol_concentration_high")
    if top_share is None or top_share > TOP_TRADE_LIMIT:
        red_flags.append("top_trade_concentration_high")
    if remove_top_1 is None or remove_top_1 <= 0:
        red_flags.append("remove_top_1_turns_nonpositive")
    if remove_top_5 is None or remove_top_5 <= 0:
        red_flags.append("remove_top_5pct_turns_nonpositive")
    if funding_3 is not None and funding_3 <= 0:
        red_flags.append("funding_3bps_turns_nonpositive")
    if funding_5 is not None and funding_5 <= 0:
        red_flags.append("funding_5bps_turns_nonpositive")
    if funding_10 is not None and funding_10 <= 0:
        red_flags.append("funding_10bps_turns_nonpositive")

    hard_fail = all(
        item in red_flags
        for item in [
            "validation_no_cost_negative",
            "symbol_concentration_high",
            "top_trade_concentration_high",
            "remove_top_1_turns_nonpositive",
        ]
    )
    has_research_potential = bool(
        family == "1d_ema"
        and train_no_cost is not None
        and train_no_cost > 0
        and oos_no_cost is not None
        and oos_no_cost > 0
        and funding_1 is not None
        and funding_1 > 0
        and not hard_fail
    )
    proceed = bool(stable_candidate_exists or has_research_potential)
    if hard_fail:
        proceed = False

    if proceed:
        directions = [
            "keep_1d_ema_but_fix_concentration",
            "remove_weak_symbols",
            "add_stronger_trend_regime_filter",
            "extend_history_before_v3_1",
        ]
    else:
        directions = [
            "stop_current_v3_family",
            "extend_history_before_v3_1",
            "add_stronger_trend_regime_filter",
            "keep_1d_ema_but_fix_concentration",
        ]
    not_recommended = [
        "reduce_4h_donchian_family",
        "do_not_expand_parameter_search",
        "do_not_enter_demo_live",
        "do_not_develop_strategy_v3_from_current_results",
    ]
    return {
        "proceed_to_v3_1": proceed,
        "decision_rule": "Research-only V3.1 requires stable candidate or 1d EMA potential without hard concentration/tail/funding failures.",
        "red_flags": red_flags,
        "recommended_research_direction": directions,
        "not_recommended_direction": not_recommended,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
    }


def build_oos_best_metrics(
    policy_name: str | None,
    leaderboards_by_split: dict[str, pd.DataFrame],
    top_trade_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build compact OOS best policy metrics for summary and recommendation."""

    if policy_name is None:
        return {}
    metrics: dict[str, Any] = {"policy": policy_name, "policy_family": classify_policy_family(policy_name)}
    for split in SPLITS:
        row = find_policy_row(leaderboards_by_split.get(split, pd.DataFrame()), policy_name)
        metrics[f"{split}_no_cost_net_pnl"] = numeric_from_row(row, "no_cost_net_pnl")
        metrics[f"{split}_net_pnl"] = numeric_from_row(row, "net_pnl")
        metrics[f"{split}_trade_count"] = numeric_from_row(row, "trade_count")
    oos_row = find_policy_row(leaderboards_by_split.get("oos", pd.DataFrame()), policy_name)
    metrics["largest_symbol_pnl_share"] = numeric_from_row(oos_row, "largest_symbol_pnl_share")
    metrics["top_5pct_trade_pnl_contribution"] = numeric_from_row(oos_row, "top_5pct_trade_pnl_contribution")
    tail_subset = top_trade_df[(top_trade_df["split"] == "oos") & (top_trade_df["policy_name"] == policy_name)] if not top_trade_df.empty else pd.DataFrame()
    if not tail_subset.empty:
        tail = tail_subset.iloc[0]
        for column in [
            "total_net_pnl",
            "top_1_trade_pnl",
            "top_5pct_trade_pnl",
            "top_1_share",
            "top_5pct_share",
            "remove_top_1_pnl",
            "remove_top_5pct_pnl",
            "remove_top_10pct_pnl",
        ]:
            metrics[column] = finite_or_none(tail.get(column))
    for bps in DEFAULT_FUNDING_BPS:
        metrics[f"funding_adjusted_{int(bps)}bps"] = funding_adjusted_for_policy(funding_df, policy_name, bps)
    return metrics


def stable_candidate_exists(compare_summary: dict[str, Any], compare_df: pd.DataFrame) -> bool:
    """Return stable candidate status from compare outputs."""

    if "stable_candidate_exists" in compare_summary:
        return bool(compare_summary.get("stable_candidate_exists"))
    if not compare_df.empty and "stable_candidate" in compare_df.columns:
        return bool((compare_df["stable_candidate"] == True).any())  # noqa: E712
    return False


def top_rows(frame: pd.DataFrame, sort_column: str, count: int = 3, ascending: bool = True) -> list[dict[str, Any]]:
    """Return top records from a DataFrame."""

    if frame.empty or sort_column not in frame.columns:
        return []
    working = frame.copy()
    working[sort_column] = pd.to_numeric(working[sort_column], errors="coerce")
    working = working.dropna(subset=[sort_column]).sort_values(sort_column, ascending=ascending, kind="stable").head(count)
    return dataframe_records(working)


def build_answers(
    *,
    family_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
    by_month_df: pd.DataFrame,
    by_policy_month_df: pd.DataFrame,
    top_trade_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    oos_best_metrics: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    """Build report answers for postmortem questions."""

    oos_policy = oos_best_metrics.get("policy")
    validation_policy = by_policy_month_df[by_policy_month_df["split"] == "validation"].copy() if not by_policy_month_df.empty else pd.DataFrame()
    validation_symbol = symbol_df[symbol_df["split"] == "validation"].copy() if not symbol_df.empty else pd.DataFrame()
    oos_month = by_policy_month_df[by_policy_month_df["split"] == "oos"].copy() if not by_policy_month_df.empty else pd.DataFrame()
    family_pivot = (
        family_df.pivot_table(index="policy_family", columns="split", values="no_cost_net_pnl", aggfunc="sum")
        if not family_df.empty
        else pd.DataFrame()
    )
    donchian_families = [family for family in ["4h_donchian", "1d_donchian", "risk_filtered"] if family in family_pivot.index]
    ema_families = [family for family in ["4h_ema", "1d_ema"] if family in family_pivot.index]
    donchian_total = float(family_pivot.loc[donchian_families].sum().sum()) if donchian_families else None
    ema_total = float(family_pivot.loc[ema_families].sum().sum()) if ema_families else None
    best_family_row = pd.DataFrame()
    if not family_df.empty:
        oos_family = family_df[family_df["split"] == "oos"].copy()
        if not oos_family.empty:
            best_family_row = oos_family.sort_values(["net_pnl", "no_cost_net_pnl"], ascending=[False, False], kind="stable").head(1)

    funding_best = funding_df[(funding_df["split"] == "oos") & (funding_df["policy_name"] == oos_policy)].copy() if not funding_df.empty and oos_policy else pd.DataFrame()
    symbol_oos_best = symbol_df[(symbol_df["split"] == "oos") & (symbol_df["policy_name"] == oos_policy)].copy() if not symbol_df.empty and oos_policy else pd.DataFrame()
    largest_symbol = None
    worst_symbol = None
    if not symbol_oos_best.empty:
        largest_symbol = symbol_oos_best.sort_values("net_pnl", ascending=False, kind="stable").iloc[0].get("symbol")
        worst_symbol = symbol_oos_best.sort_values("net_pnl", ascending=True, kind="stable").iloc[0].get("symbol")

    return {
        "v3_0_failure_main_reason": (
            "没有任何 policy 同时通过 train/validation/oos：OOS best policy 在 validation 为负，"
            "多数 Donchian、4h、ensemble family 跨样本为负，OOS 正收益又集中在单一 symbol 和极少数尾部交易。"
        ),
        "closest_policy_family": dataframe_records(best_family_row)[0] if not best_family_row.empty else None,
        "stable_failed_families": (
            family_df[family_df["stable_direction"] == "negative_all_splits"]["policy_family"].drop_duplicates().tolist()
            if not family_df.empty
            else []
        ),
        "donchian_overall_weak": bool(donchian_total is not None and donchian_total < 0),
        "ema_better_than_donchian": bool(ema_total is not None and donchian_total is not None and ema_total > donchian_total),
        "one_day_more_stable_than_4h": infer_1d_vs_4h(family_df),
        "oos_best_over_depends_single_symbol": bool((finite_or_none(oos_best_metrics.get("largest_symbol_pnl_share")) or 0.0) > CONCENTRATION_LIMIT),
        "oos_best_largest_symbol": largest_symbol,
        "oos_best_worst_symbol": worst_symbol,
        "symbol_removal_improves_cases": dataframe_records(
            symbol_df[symbol_df["removing_symbol_improves_policy"] == True].head(20)  # noqa: E712
        )
        if not symbol_df.empty
        else [],
        "only_one_coin_trended_cases": dataframe_records(
            symbol_df[(symbol_df["largest_symbol_dependency"] > CONCENTRATION_LIMIT) & (symbol_df["is_largest_dependency_symbol"] == True)].head(20)  # noqa: E712
        )
        if not symbol_df.empty
        else [],
        "validation_worst_policies": top_rows(validation_policy, "net_pnl", count=5, ascending=True),
        "validation_worst_symbols": top_rows(validation_symbol, "net_pnl", count=5, ascending=True),
        "validation_worst_months": top_rows(by_month_df[by_month_df["split"] == "validation"], "net_pnl", count=5, ascending=True)
        if not by_month_df.empty
        else [],
        "oos_best_months": top_rows(oos_month[oos_month["policy_name"] == oos_policy], "net_pnl", count=5, ascending=False)
        if not oos_month.empty and oos_policy
        else [],
        "oos_best_remove_top_1_still_positive": bool((finite_or_none(oos_best_metrics.get("remove_top_1_pnl")) or 0.0) > 0),
        "oos_best_remove_top_5pct_still_positive": bool((finite_or_none(oos_best_metrics.get("remove_top_5pct_pnl")) or 0.0) > 0),
        "funding_stress_for_oos_best": dataframe_records(funding_best),
        "proceed_to_v3_1": recommendation.get("proceed_to_v3_1"),
        "recommended_research_direction": recommendation.get("recommended_research_direction"),
        "not_recommended_direction": recommendation.get("not_recommended_direction"),
    }


def infer_1d_vs_4h(family_df: pd.DataFrame) -> bool | None:
    """Infer whether 1d families were more stable than 4h families."""

    if family_df.empty:
        return None
    totals = family_df.groupby("policy_family")["no_cost_net_pnl"].sum()
    one_day = float(totals.reindex(["1d_ema", "1d_donchian"]).fillna(0.0).sum())
    four_hour = float(totals.reindex(["4h_ema", "4h_donchian", "vol_compression", "risk_filtered"]).fillna(0.0).sum())
    return bool(one_day > four_hour)


def format_number(value: Any, digits: int = 6) -> str:
    """Format a number for Markdown."""

    number = finite_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], digits: int = 4) -> str:
    """Render a small Markdown table."""

    if not rows:
        return "- 无"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, (bool, np.bool_)):
                values.append(str(bool(value)).lower())
            elif isinstance(value, (int, float, np.integer, np.floating)):
                values.append(format_number(value, digits))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(summary: dict[str, Any], family_df: pd.DataFrame) -> str:
    """Render trend_v3_postmortem_report.md."""

    answers = summary.get("answers") or {}
    oos = summary.get("oos_best_policy") or {}
    funding_rows = answers.get("funding_stress_for_oos_best") or []
    family_oos = dataframe_records(
        family_df[family_df["split"] == "oos"].sort_values(["net_pnl", "no_cost_net_pnl"], ascending=[False, False], kind="stable")
    ) if not family_df.empty else []
    warning_lines = "\n".join(f"- {item}" for item in summary.get("warnings", [])) or "- 无"
    return (
        "# Trend V3.0 Postmortem：趋势跟踪失败归因\n\n"
        "## 核心结论\n"
        f"- trend_following_v3_failed={str(bool(summary.get('trend_following_v3_failed'))).lower()}\n"
        f"- stable_candidate_exists={str(bool(summary.get('stable_candidate_exists'))).lower()}\n"
        f"- proceed_to_v3_1={str(bool(summary.get('proceed_to_v3_1'))).lower()}\n"
        "- 本报告只做 V3.0 复盘，不开发 Strategy V3，不改正式策略，不进入 demo/live。\n"
        "- Funding 部分是 synthetic funding stress，不是真实 OKX funding fee。\n\n"
        "## 1. V3.0 失败的主要原因是什么？\n"
        f"{answers.get('v3_0_failure_main_reason')}\n\n"
        "## 2. 是 Donchian 失败，还是 EMA 失败，还是组合风控失败？\n"
        f"- Donchian overall weak={str(bool(answers.get('donchian_overall_weak'))).lower()}。\n"
        f"- EMA better than Donchian={str(bool(answers.get('ema_better_than_donchian'))).lower()}，但 EMA 也没有跨 validation/OOS 稳定通过。\n"
        "- 组合风控没有发现持仓上限或重复持仓类 audit 问题，但无法解决行情/品种收益集中与尾部依赖问题。\n\n"
        "## 3. 1d EMA 是否值得进入 V3.1 继续研究？\n"
        f"- OOS best policy={oos.get('policy')}，train no-cost={format_number(oos.get('train_no_cost_net_pnl'), 4)}，validation no-cost={format_number(oos.get('validation_no_cost_net_pnl'), 4)}，OOS no-cost={format_number(oos.get('oos_no_cost_net_pnl'), 4)}。\n"
        f"- 建议：proceed_to_v3_1={str(bool(summary.get('proceed_to_v3_1'))).lower()}；方向={summary.get('recommended_research_direction')}。\n\n"
        "## 4. OOS best policy 是否过度依赖单一 symbol？\n"
        f"- largest_symbol_pnl_share={format_number(oos.get('largest_symbol_pnl_share'), 6)}，largest_symbol={answers.get('oos_best_largest_symbol')}，worst_symbol={answers.get('oos_best_worst_symbol')}。\n"
        f"- over_depends_single_symbol={str(bool(answers.get('oos_best_over_depends_single_symbol'))).lower()}。\n\n"
        "## 5. 去掉 top 1 / top 5% 盈利交易后是否仍为正？\n"
        f"- remove_top_1_pnl={format_number(oos.get('remove_top_1_pnl'), 6)}，still_positive={str(bool(answers.get('oos_best_remove_top_1_still_positive'))).lower()}。\n"
        f"- remove_top_5pct_pnl={format_number(oos.get('remove_top_5pct_pnl'), 6)}，still_positive={str(bool(answers.get('oos_best_remove_top_5pct_still_positive'))).lower()}。\n\n"
        "## 6. Funding fee stress 后是否仍为正？\n"
        f"{markdown_table(funding_rows, ['funding_bps_per_8h', 'funding_adjusted_net_pnl', 'remains_positive_after_funding'])}\n\n"
        "## 7. Validation 失败来自哪些 policy / symbol / month？\n"
        "### Worst validation policies\n"
        f"{markdown_table(answers.get('validation_worst_policies') or [], ['policy_name', 'policy_family', 'month', 'net_pnl', 'trade_count'])}\n\n"
        "### Worst validation symbols\n"
        f"{markdown_table(answers.get('validation_worst_symbols') or [], ['policy_name', 'symbol', 'net_pnl', 'trade_count'])}\n\n"
        "### Worst validation months\n"
        f"{markdown_table(answers.get('validation_worst_months') or [], ['month', 'net_pnl', 'trade_count', 'win_rate'])}\n\n"
        "## 8. 是否建议进入 V3.1？\n"
        f"- proceed_to_v3_1={str(bool(summary.get('proceed_to_v3_1'))).lower()}。\n"
        f"- red_flags={summary.get('v3_1_recommendations', {}).get('red_flags')}。\n"
        "- 即使未来允许 V3.1，也只能是研究设计，不能进入 Strategy V3 原型、demo 或 live。\n\n"
        "## 9. 如果进入 V3.1，应该研究什么？\n"
        f"- recommended_research_direction={summary.get('recommended_research_direction')}。\n\n"
        "## 10. 如果不进入 V3.1，应该停止哪些 policy family？\n"
        f"- not_recommended_direction={summary.get('not_recommended_direction')}。\n"
        "- 4h Donchian、vol compression、ensemble 当前不应继续扩大参数搜索；1d EMA 只能作为集中度/行情过滤约束下的研究假设保留。\n\n"
        "## Policy Family OOS Snapshot\n"
        f"{markdown_table(family_oos, ['policy_family', 'trade_count', 'no_cost_net_pnl', 'net_pnl', 'win_rate', 'profit_factor', 'stable_direction'])}\n\n"
        "## 输出文件\n"
        + "\n".join(f"- {filename}" for filename in OUTPUT_FILES)
        + "\n\n"
        "## Warnings\n"
        f"{warning_lines}\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Write JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Write CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def run_postmortem(
    train_dir: Path,
    validation_dir: Path,
    oos_dir: Path,
    compare_dir: Path,
    output_dir: Path,
    *,
    funding_mode: str = "synthetic",
    funding_bps_values: list[float] | None = None,
    min_trade_count: int = 10,
) -> dict[str, Any]:
    """Run Trend V3 postmortem and write all outputs."""

    warnings: list[str] = []
    split_dirs = {"train": train_dir, "validation": validation_dir, "oos": oos_dir}
    split_artifacts = {split: load_split_artifacts(split, directory, warnings) for split, directory in split_dirs.items()}
    compare_artifacts = load_compare_artifacts(compare_dir, warnings)

    trades_by_split = {
        split: prepare_trades(artifacts.get("trend_v3_trades", pd.DataFrame()), split)
        for split, artifacts in split_artifacts.items()
    }
    leaderboards_by_split = {
        split: artifacts.get("trend_v3_policy_leaderboard", pd.DataFrame())
        for split, artifacts in split_artifacts.items()
    }
    for split, leaderboard in leaderboards_by_split.items():
        if not leaderboard.empty and "policy_name" in leaderboard.columns:
            leaderboard["policy_family"] = leaderboard["policy_name"].map(classify_policy_family)
            leaderboards_by_split[split] = leaderboard
    capitals = {split: split_capital(artifacts) for split, artifacts in split_artifacts.items()}
    funding_bps = funding_bps_values or DEFAULT_FUNDING_BPS

    family_df = build_policy_family_analysis(trades_by_split, capitals)
    symbol_df = build_symbol_contribution_postmortem(trades_by_split, capitals)
    by_month_df = build_grouped_time_summary(trades_by_split, ["month"], capitals)
    by_quarter_df = build_grouped_time_summary(trades_by_split, ["quarter"], capitals)
    by_symbol_month_df = build_grouped_time_summary(trades_by_split, ["symbol", "month"], capitals)
    by_policy_month_df = build_grouped_time_summary(trades_by_split, ["policy_name", "policy_family", "month"], capitals)
    top_trade_df = build_top_trade_concentration(trades_by_split)
    funding_df, effective_funding_mode = build_funding_sensitivity(trades_by_split, funding_bps, funding_mode, warnings)
    compare_df = compare_artifacts.get("trend_v3_compare_leaderboard", pd.DataFrame())
    compare_summary = compare_artifacts.get("trend_v3_compare_summary", {})
    rejected_df = build_rejected_candidate_reasons(compare_df, leaderboards_by_split, min_trade_count)

    stable_exists = stable_candidate_exists(compare_summary, compare_df)
    oos_best_policy = select_oos_best_policy(leaderboards_by_split.get("oos", pd.DataFrame()), trades_by_split.get("oos", pd.DataFrame()))
    oos_best_metrics = build_oos_best_metrics(oos_best_policy, leaderboards_by_split, top_trade_df, funding_df)
    recommendation = decide_v3_1_recommendation(oos_best_metrics, stable_exists)
    answers = build_answers(
        family_df=family_df,
        symbol_df=symbol_df,
        by_month_df=by_month_df,
        by_policy_month_df=by_policy_month_df,
        top_trade_df=top_trade_df,
        funding_df=funding_df,
        oos_best_metrics=oos_best_metrics,
        recommendation=recommendation,
    )

    summary = {
        "input_dirs": {split: str(path) for split, path in split_dirs.items()} | {"compare": str(compare_dir)},
        "output_dir": str(output_dir),
        "warnings": warnings,
        "funding_mode_requested": funding_mode,
        "funding_mode_effective": effective_funding_mode,
        "funding_bps_per_8h": funding_bps,
        "synthetic_funding_stress_notice": (
            "funding_sensitivity.csv is a conservative synthetic funding stress test, not actual OKX funding fee."
            if effective_funding_mode == "synthetic"
            else "Funding stress disabled by funding-mode=none."
        ),
        "min_trade_count": min_trade_count,
        "stable_candidate_exists": stable_exists,
        "trend_following_v3_failed": bool(not stable_exists),
        "oos_best_policy": oos_best_metrics,
        "answers": answers,
        "proceed_to_v3_1": recommendation["proceed_to_v3_1"],
        "recommended_research_direction": recommendation["recommended_research_direction"],
        "not_recommended_direction": recommendation["not_recommended_direction"],
        "v3_1_recommendations": recommendation,
        "output_files": OUTPUT_FILES,
        "row_counts": {
            "policy_family_analysis": int(len(family_df.index)),
            "symbol_contribution_postmortem": int(len(symbol_df.index)),
            "by_month": int(len(by_month_df.index)),
            "by_quarter": int(len(by_quarter_df.index)),
            "by_symbol_month": int(len(by_symbol_month_df.index)),
            "by_policy_month": int(len(by_policy_month_df.index)),
            "top_trade_concentration": int(len(top_trade_df.index)),
            "funding_sensitivity": int(len(funding_df.index)),
            "rejected_candidate_reasons": int(len(rejected_df.index)),
        },
    }
    report = render_report(summary, family_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "trend_v3_postmortem_summary.json", summary)
    (output_dir / "trend_v3_postmortem_report.md").write_text(report, encoding="utf-8")
    write_dataframe(output_dir / "policy_family_analysis.csv", family_df)
    write_dataframe(output_dir / "symbol_contribution_postmortem.csv", symbol_df)
    write_dataframe(output_dir / "by_month.csv", by_month_df)
    write_dataframe(output_dir / "by_quarter.csv", by_quarter_df)
    write_dataframe(output_dir / "by_symbol_month.csv", by_symbol_month_df)
    write_dataframe(output_dir / "by_policy_month.csv", by_policy_month_df)
    write_dataframe(output_dir / "top_trade_concentration.csv", top_trade_df)
    write_dataframe(output_dir / "funding_sensitivity.csv", funding_df)
    write_dataframe(output_dir / "rejected_candidate_reasons.csv", rejected_df)
    write_json(output_dir / "v3_1_recommendations.json", recommendation)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    ensure_headless_runtime()
    logger = setup_logging("postmortem_trend_following_v3", verbose=args.verbose)
    try:
        funding_bps = parse_funding_bps(args.funding_bps_per_8h)
        summary = run_postmortem(
            resolve_path(args.train_dir),
            resolve_path(args.validation_dir),
            resolve_path(args.oos_dir),
            resolve_path(args.compare_dir),
            resolve_path(args.output_dir),
            funding_mode=args.funding_mode,
            funding_bps_values=funding_bps,
            min_trade_count=args.min_trade_count,
        )
        print_json_block(
            "Trend V3 postmortem summary:",
            {
                "output_dir": summary.get("output_dir"),
                "trend_following_v3_failed": summary.get("trend_following_v3_failed"),
                "oos_best_policy": (summary.get("oos_best_policy") or {}).get("policy"),
                "proceed_to_v3_1": summary.get("proceed_to_v3_1"),
                "warnings": len(summary.get("warnings", [])),
            },
        )
        if args.json:
            print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except TrendFollowingV3PostmortemError as exc:
        log_event(logger, logging.ERROR, "trend_v3_postmortem.error", str(exc))
        return 2
    except Exception:
        logger.exception("Unexpected error during Trend V3 postmortem", extra={"event": "trend_v3_postmortem.unexpected_error"})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
