#!/usr/bin/env python3
"""Run a headless vn.py CTA backtest against local OKX 1m bar data."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common_runtime import PROJECT_ROOT, ensure_headless_runtime, log_event, print_json_block, setup_logging, to_jsonable
from history_time_utils import DEFAULT_TIMEZONE, HistoryRange, parse_history_range
from history_utils import get_database_timezone, parse_interval
from verify_okx_history import build_payload as build_data_check_payload, verify_history_range


STRATEGY_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "strategy_default.json"
INSTRUMENT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "instruments" / "btcusdt_swap_okx.json"
DEFAULT_VT_SYMBOL: str = "BTCUSDT_SWAP_OKX.GLOBAL"
WARNING_TEXT: str = (
    "标准 vn.py CTA BacktestingEngine 未自动计入 OKX perpetual funding fee，"
    "本结果仅包含价格盈亏、手续费、滑点等常规项。"
)


class ConfigurationError(Exception):
    """Raised when local files or CLI parameters are invalid."""


class BacktestError(Exception):
    """Raised when backtest execution fails."""


@dataclass(frozen=True, slots=True)
class InstrumentMeta:
    """Instrument metadata used to parameterize the backtest."""

    vt_symbol: str
    symbol: str
    exchange: str
    name: str
    size: float
    pricetick: float
    min_volume: float


@dataclass(frozen=True, slots=True)
class RoundTripStats:
    """Derived round-trip statistics from trade records."""

    closed_trade_count: int
    win_rate: float | None
    profit_loss_ratio: float | None
    average_win: float | None
    average_loss: float | None
    gross_profit: float
    gross_loss: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run headless vn.py BacktestingEngine for the OKX MHF CTA strategy."
    )
    parser.add_argument(
        "--strategy-config",
        default=str(STRATEGY_CONFIG_PATH.relative_to(PROJECT_ROOT)),
        help="Strategy config JSON path. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--setting-overrides",
        help="Inline JSON object merged into strategy_config.setting. CLI overrides take precedence.",
    )
    parser.add_argument(
        "--setting-overrides-file",
        help="JSON file merged into strategy_config.setting before --setting-overrides.",
    )
    parser.add_argument(
        "--vt-symbol",
        default=DEFAULT_VT_SYMBOL,
        help=f"Backtest vt_symbol. Default: {DEFAULT_VT_SYMBOL}.",
    )
    parser.add_argument(
        "--allow-bankrupt-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Still export diagnostics/artifacts when balance drops to 0 or below. Default: enabled.",
    )
    parser.add_argument(
        "--mode",
        choices=("bar",),
        default="bar",
        help="Backtesting mode. Only bar is supported here.",
    )
    parser.add_argument(
        "--start",
        default="2025-01-01",
        help="Backtest start datetime in ISO format. Date-only means 00:00:00 in --timezone.",
    )
    parser.add_argument(
        "--end",
        default="2026-03-31",
        help="Backtest end datetime in ISO format. Date-only means the whole natural day in --timezone.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"History timezone used for verify/preflight. Default: {DEFAULT_TIMEZONE}.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=5000.0,
        help="Backtest initial capital. Default: 5000.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0005,
        help="Commission rate. Default: 0.0005.",
    )
    parser.add_argument(
        "--slippage-mode",
        choices=("ticks", "absolute"),
        default="ticks",
        help="Interpret --slippage as ticks or absolute price. Default: ticks.",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=2.0,
        help="Slippage value. Default: 2.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional custom output directory. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip sqlite history completeness check before backtesting.",
    )
    parser.add_argument(
        "--data-check-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast when sqlite history still has gaps. Default: enabled.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def resolve_path_arg(path_arg: str | None, default_path: Path) -> Path:
    """Resolve a CLI path argument relative to project root."""

    if not path_arg:
        return default_path

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def configure_sqlite_settings(logger: logging.Logger) -> None:
    """Force vn.py to use the project-local sqlite database."""

    from vnpy.trader.setting import SETTINGS

    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = "database.db"

    log_event(
        logger,
        logging.INFO,
        "backtest.db_settings",
        "Configured vn.py database settings for sqlite",
        database_name=SETTINGS["database.name"],
        database_database=SETTINGS["database.database"],
    )


def load_json_file(path: Path) -> dict[str, Any]:
    """Load JSON config from disk."""

    if not path.exists():
        raise ConfigurationError(f"配置文件不存在: {path}")

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigurationError(f"读取 JSON 配置失败: {path} | {exc!r}") from exc

    if not isinstance(content, dict):
        raise ConfigurationError(f"JSON 顶层结构必须是对象: {path}")
    return content


def load_setting_overrides(
    overrides_file_arg: str | None,
    overrides_arg: str | None,
) -> tuple[dict[str, Any], Path | None]:
    """Load and merge setting overrides from file and inline JSON."""

    merged_overrides: dict[str, Any] = {}
    overrides_file_path: Path | None = None

    if overrides_file_arg:
        overrides_file_path = resolve_path_arg(overrides_file_arg, PROJECT_ROOT)
        overrides_from_file = load_json_file(overrides_file_path)
        merged_overrides.update(overrides_from_file)

    if overrides_arg:
        try:
            inline_overrides = json.loads(overrides_arg)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"--setting-overrides 不是合法 JSON: {exc.msg}") from exc

        if not isinstance(inline_overrides, dict):
            raise ConfigurationError("--setting-overrides 顶层结构必须是 JSON 对象")

        merged_overrides.update(inline_overrides)

    return merged_overrides, overrides_file_path


def load_instrument_meta(path: Path) -> InstrumentMeta:
    """Load instrument metadata from config/instruments."""

    content = load_json_file(path)

    try:
        return InstrumentMeta(
            vt_symbol=str(content["vt_symbol"]),
            symbol=str(content["symbol"]),
            exchange=str(content["exchange"]),
            name=str(content.get("name", "")),
            size=float(content.get("size", content.get("contract_size", 0.0)) or 0.0),
            pricetick=float(content.get("pricetick", 0.0) or 0.0),
            min_volume=float(content.get("min_volume", 0.0) or 0.0),
        )
    except KeyError as exc:
        raise ConfigurationError(f"合约元数据缺少字段: {exc.args[0]} | {path}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"合约元数据字段类型错误: {path} | {exc!r}") from exc


def resolve_strategy_class(class_name: str) -> type:
    """Resolve strategy class from the local strategies package."""

    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    try:
        module = importlib.import_module("strategies")
    except Exception as exc:
        raise ConfigurationError(f"无法导入 strategies 包: {exc!r}") from exc

    try:
        strategy_class = getattr(module, class_name)
    except AttributeError as exc:
        raise ConfigurationError(f"在 strategies 包中找不到策略类: {class_name}") from exc

    return strategy_class


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    """Resolve final report output directory."""

    if output_dir_arg:
        path = Path(output_dir_arg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "reports" / "backtest" / timestamp


def resolve_slippage(pricetick: float, slippage_mode: str, slippage: float) -> float:
    """Resolve absolute slippage for BacktestingEngine."""

    if slippage < 0:
        raise ConfigurationError(f"slippage 不能小于 0，当前值为: {slippage}")

    if slippage_mode == "absolute":
        return slippage

    if pricetick <= 0:
        raise ConfigurationError("slippage-mode=ticks 时，pricetick 必须大于 0")
    return slippage * pricetick


def fill_strategy_setting(
    raw_setting: dict[str, Any],
    instrument_meta: InstrumentMeta,
    capital: float,
    logger: logging.Logger,
    setting_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill strategy defaults from instrument metadata when JSON still contains zero placeholders."""

    setting = dict(raw_setting)
    replacements: dict[str, float] = {}

    if setting_overrides:
        setting.update(setting_overrides)
        log_event(
            logger,
            logging.INFO,
            "backtest.strategy_setting_override",
            "Applied temporary strategy setting overrides",
            setting_overrides=setting_overrides,
        )

    if float(setting.get("contract_size", 0.0) or 0.0) <= 0:
        setting["contract_size"] = instrument_meta.size
        replacements["contract_size"] = instrument_meta.size

    if float(setting.get("min_volume", 0.0) or 0.0) <= 0:
        setting["min_volume"] = instrument_meta.min_volume
        replacements["min_volume"] = instrument_meta.min_volume

    if float(setting.get("pricetick", 0.0) or 0.0) <= 0:
        setting["pricetick"] = instrument_meta.pricetick
        replacements["pricetick"] = instrument_meta.pricetick

    if "capital_per_strategy" in setting:
        previous_capital = float(setting.get("capital_per_strategy", 0.0) or 0.0)
        if previous_capital != capital:
            setting["capital_per_strategy"] = capital
            replacements["capital_per_strategy"] = capital

    if replacements:
        log_event(
            logger,
            logging.INFO,
            "backtest.strategy_setting_filled",
            "Filled strategy placeholders from instrument metadata / CLI capital",
            replacements=replacements,
        )

    return setting


def validate_runtime_config(
    vt_symbol: str,
    instrument_meta: InstrumentMeta,
    start: datetime,
    end: datetime,
) -> None:
    """Validate backtest runtime assumptions."""

    if start >= end:
        raise ConfigurationError(f"开始时间必须小于结束时间: start={start}, end={end}")

    if vt_symbol != instrument_meta.vt_symbol:
        raise ConfigurationError(
            f"当前仓库的合约元数据只覆盖 {instrument_meta.vt_symbol}，"
            f"收到 --vt-symbol={vt_symbol}。当前最小工程先只支持该标的。"
        )

    if instrument_meta.size <= 0:
        raise ConfigurationError(f"合约元数据 size 无效: {instrument_meta.size}")
    if instrument_meta.pricetick <= 0:
        raise ConfigurationError(f"合约元数据 pricetick 无效: {instrument_meta.pricetick}")
    if instrument_meta.min_volume <= 0:
        raise ConfigurationError(f"合约元数据 min_volume 无效: {instrument_meta.min_volume}")


def make_engine_output(logger: logging.Logger):
    """Create a logger-backed output callback for BacktestingEngine."""

    def _output(message: str) -> None:
        log_event(
            logger,
            logging.INFO,
            "backtest.engine",
            message,
        )

    return _output


def write_json_file(path: Path, payload: Any) -> None:
    """Write JSON data with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Export DataFrame to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, encoding="utf-8", index=False)


def prepare_daily_pnl_dataframe(
    daily_df: pd.DataFrame,
    initial_capital: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a copy of daily pnl data with manual balance/drawdown columns when possible."""

    prepared_df = daily_df.copy()
    missing_columns: list[str] = []

    if prepared_df.empty:
        return prepared_df, missing_columns

    if "net_pnl" in prepared_df.columns:
        prepared_df["balance"] = prepared_df["net_pnl"].cumsum() + float(initial_capital)
    elif "balance" not in prepared_df.columns:
        missing_columns.append("net_pnl")

    if "balance" in prepared_df.columns:
        balance_series = pd.to_numeric(prepared_df["balance"], errors="coerce")
        pre_balance = balance_series.shift(1)
        if len(pre_balance.index):
            pre_balance.iloc[0] = float(initial_capital)
        ratio = balance_series / pre_balance.replace(0, np.nan)
        ratio[ratio <= 0] = np.nan
        prepared_df["balance"] = balance_series
        prepared_df["return"] = np.log(ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prepared_df["highlevel"] = balance_series.cummax()
        prepared_df["drawdown"] = balance_series - prepared_df["highlevel"]
        prepared_df["ddpercent"] = (
            prepared_df["drawdown"] / prepared_df["highlevel"].replace(0, np.nan) * 100.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        missing_columns.append("balance")

    return prepared_df, sorted(set(missing_columns))


def _sum_numeric_column(df: pd.DataFrame, column_name: str) -> float | None:
    """Safely sum a numeric DataFrame column."""

    if column_name not in df.columns:
        return None
    return float(pd.to_numeric(df[column_name], errors="coerce").fillna(0.0).sum())


def _pick_extreme_day(df: pd.DataFrame, column_name: str, pick_min: bool) -> dict[str, Any] | None:
    """Pick one best/worst day summary from a DataFrame."""

    if column_name not in df.columns or df.empty:
        return None

    series = pd.to_numeric(df[column_name], errors="coerce")
    if series.dropna().empty:
        return None

    index_label = series.idxmin() if pick_min else series.idxmax()
    row = df.loc[index_label]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    date_value: Any = index_label
    if hasattr(index_label, "isoformat"):
        date_value = index_label.isoformat()

    return {
        "date": date_value,
        "net_pnl": float(series.loc[index_label]),
        "trade_count": float(pd.to_numeric(pd.Series([row.get("trade_count", 0)]), errors="coerce").fillna(0.0).iloc[0]),
        "balance": float(pd.to_numeric(pd.Series([row.get("balance", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
        if "balance" in row.index
        else None,
    }


def analyze_daily_pnl(daily_df: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    """Build backtest diagnostics directly from daily pnl output without relying on vn.py statistics."""

    prepared_df, missing_columns = prepare_daily_pnl_dataframe(daily_df, initial_capital)
    diagnostics: dict[str, Any] = {
        "bankrupt": False,
        "first_bankrupt_date": None,
        "final_balance": None,
        "min_balance": None,
        "max_balance": None,
        "total_net_pnl_sum": None,
        "total_commission": None,
        "total_slippage": None,
        "total_turnover": None,
        "daily_trade_count_sum": None,
        "worst_day": None,
        "best_day": None,
        "max_daily_loss": None,
        "max_daily_profit": None,
        "max_drawdown_abs_manual": None,
        "max_drawdown_pct_manual": None,
        "first_balance_below_50pct_date": None,
        "first_balance_below_20pct_date": None,
        "missing_columns": missing_columns,
        "daily_row_count": int(len(prepared_df)),
        "initial_capital": float(initial_capital),
    }

    if prepared_df.empty:
        diagnostics["missing_columns"] = sorted(set(missing_columns + ["daily_df_empty"]))
        return diagnostics

    if "balance" in prepared_df.columns:
        balance_series = pd.to_numeric(prepared_df["balance"], errors="coerce")
        valid_balance = balance_series.dropna()
        if not valid_balance.empty:
            diagnostics["final_balance"] = float(valid_balance.iloc[-1])
            diagnostics["min_balance"] = float(valid_balance.min())
            diagnostics["max_balance"] = float(valid_balance.max())

            bankrupt_mask = valid_balance <= 0
            if bool(bankrupt_mask.any()):
                diagnostics["bankrupt"] = True
                first_bankrupt_index = bankrupt_mask[bankrupt_mask].index[0]
                diagnostics["first_bankrupt_date"] = (
                    first_bankrupt_index.isoformat()
                    if hasattr(first_bankrupt_index, "isoformat")
                    else str(first_bankrupt_index)
                )

            balance_below_50 = valid_balance[valid_balance <= float(initial_capital) * 0.5]
            if not balance_below_50.empty:
                first_below_50 = balance_below_50.index[0]
                diagnostics["first_balance_below_50pct_date"] = (
                    first_below_50.isoformat() if hasattr(first_below_50, "isoformat") else str(first_below_50)
                )

            balance_below_20 = valid_balance[valid_balance <= float(initial_capital) * 0.2]
            if not balance_below_20.empty:
                first_below_20 = balance_below_20.index[0]
                diagnostics["first_balance_below_20pct_date"] = (
                    first_below_20.isoformat() if hasattr(first_below_20, "isoformat") else str(first_below_20)
                )
    else:
        diagnostics["missing_columns"] = sorted(set(diagnostics["missing_columns"] + ["balance"]))

    diagnostics["total_net_pnl_sum"] = _sum_numeric_column(prepared_df, "net_pnl")
    diagnostics["total_commission"] = _sum_numeric_column(prepared_df, "commission")
    diagnostics["total_slippage"] = _sum_numeric_column(prepared_df, "slippage")
    diagnostics["total_turnover"] = _sum_numeric_column(prepared_df, "turnover")
    diagnostics["daily_trade_count_sum"] = _sum_numeric_column(prepared_df, "trade_count")

    if "net_pnl" in prepared_df.columns:
        pnl_series = pd.to_numeric(prepared_df["net_pnl"], errors="coerce").dropna()
        if not pnl_series.empty:
            diagnostics["max_daily_loss"] = float(pnl_series.min())
            diagnostics["max_daily_profit"] = float(pnl_series.max())
        diagnostics["worst_day"] = _pick_extreme_day(prepared_df, "net_pnl", pick_min=True)
        diagnostics["best_day"] = _pick_extreme_day(prepared_df, "net_pnl", pick_min=False)
    else:
        diagnostics["missing_columns"] = sorted(set(diagnostics["missing_columns"] + ["net_pnl"]))

    if "drawdown" in prepared_df.columns:
        diagnostics["max_drawdown_abs_manual"] = float(
            pd.to_numeric(prepared_df["drawdown"], errors="coerce").fillna(0.0).min()
        )
    else:
        diagnostics["missing_columns"] = sorted(set(diagnostics["missing_columns"] + ["drawdown"]))

    if "ddpercent" in prepared_df.columns:
        diagnostics["max_drawdown_pct_manual"] = float(
            pd.to_numeric(prepared_df["ddpercent"], errors="coerce").fillna(0.0).min()
        )
    else:
        diagnostics["missing_columns"] = sorted(set(diagnostics["missing_columns"] + ["ddpercent"]))

    return diagnostics


def trade_to_record(trade: Any) -> dict[str, Any]:
    """Convert TradeData into a flat CSV-ready record."""

    return {
        "datetime": getattr(trade, "datetime", None),
        "vt_tradeid": getattr(trade, "vt_tradeid", ""),
        "vt_orderid": getattr(trade, "vt_orderid", ""),
        "symbol": getattr(trade, "symbol", ""),
        "exchange": getattr(getattr(trade, "exchange", None), "value", getattr(trade, "exchange", "")),
        "direction": getattr(getattr(trade, "direction", None), "value", getattr(trade, "direction", "")),
        "offset": getattr(getattr(trade, "offset", None), "value", getattr(trade, "offset", "")),
        "price": getattr(trade, "price", 0.0),
        "volume": getattr(trade, "volume", 0.0),
        "gateway_name": getattr(trade, "gateway_name", ""),
        "tradeid": getattr(trade, "tradeid", ""),
        "orderid": getattr(trade, "orderid", ""),
    }


def order_to_record(order: Any) -> dict[str, Any]:
    """Convert OrderData into a flat CSV-ready record."""

    return {
        "datetime": getattr(order, "datetime", None),
        "vt_orderid": getattr(order, "vt_orderid", ""),
        "symbol": getattr(order, "symbol", ""),
        "exchange": getattr(getattr(order, "exchange", None), "value", getattr(order, "exchange", "")),
        "type": getattr(getattr(order, "type", None), "value", getattr(order, "type", "")),
        "direction": getattr(getattr(order, "direction", None), "value", getattr(order, "direction", "")),
        "offset": getattr(getattr(order, "offset", None), "value", getattr(order, "offset", "")),
        "price": getattr(order, "price", 0.0),
        "volume": getattr(order, "volume", 0.0),
        "traded": getattr(order, "traded", 0.0),
        "status": getattr(getattr(order, "status", None), "value", getattr(order, "status", "")),
        "gateway_name": getattr(order, "gateway_name", ""),
        "orderid": getattr(order, "orderid", ""),
        "reference": getattr(order, "reference", ""),
    }


def export_records_csv(records: list[dict[str, Any]], output_path: Path, columns: list[str]) -> None:
    """Export list-of-dicts records to CSV with stable columns."""

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in df.columns:
                df[column] = None
        df = df[columns]

    export_dataframe_csv(df, output_path)


def compute_round_trip_stats(trades: list[Any], contract_size: float) -> RoundTripStats:
    """Compute win rate and profit/loss ratio from ordered trade fills."""

    if contract_size <= 0:
        return RoundTripStats(
            closed_trade_count=0,
            win_rate=None,
            profit_loss_ratio=None,
            average_win=None,
            average_loss=None,
            gross_profit=0.0,
            gross_loss=0.0,
        )

    sorted_trades = sorted(
        trades,
        key=lambda trade: (
            getattr(trade, "datetime", datetime.min),
            getattr(trade, "tradeid", ""),
            getattr(trade, "orderid", ""),
        ),
    )

    position: float = 0.0
    avg_entry_price: float = 0.0
    current_round_trip_pnl: float = 0.0
    round_trip_pnls: list[float] = []
    epsilon: float = 1e-12

    for trade in sorted_trades:
        direction_value = getattr(getattr(trade, "direction", None), "name", "")
        trade_price = float(getattr(trade, "price", 0.0) or 0.0)
        trade_volume = float(getattr(trade, "volume", 0.0) or 0.0)
        if trade_volume <= 0:
            continue

        signed_qty = trade_volume if direction_value == "LONG" else -trade_volume
        remaining_qty = signed_qty

        while abs(remaining_qty) > epsilon:
            if abs(position) <= epsilon or position * remaining_qty > 0:
                new_position = position + remaining_qty
                if abs(position) <= epsilon:
                    avg_entry_price = trade_price
                else:
                    avg_entry_price = (
                        abs(position) * avg_entry_price + abs(remaining_qty) * trade_price
                    ) / abs(new_position)

                position = new_position
                remaining_qty = 0.0
                continue

            closing_qty = min(abs(position), abs(remaining_qty))
            if position > 0:
                realized_pnl = (trade_price - avg_entry_price) * closing_qty * contract_size
                position -= closing_qty
                remaining_qty += closing_qty
            else:
                realized_pnl = (avg_entry_price - trade_price) * closing_qty * contract_size
                position += closing_qty
                remaining_qty -= closing_qty

            current_round_trip_pnl += realized_pnl

            if abs(position) <= epsilon:
                position = 0.0
                avg_entry_price = 0.0
                round_trip_pnls.append(current_round_trip_pnl)
                current_round_trip_pnl = 0.0

    wins = [pnl for pnl in round_trip_pnls if pnl > 0]
    losses = [pnl for pnl in round_trip_pnls if pnl < 0]
    closed_trade_count = len(round_trip_pnls)

    if closed_trade_count == 0:
        return RoundTripStats(
            closed_trade_count=0,
            win_rate=None,
            profit_loss_ratio=None,
            average_win=None,
            average_loss=None,
            gross_profit=0.0,
            gross_loss=0.0,
        )

    win_rate = len(wins) / closed_trade_count * 100
    average_win = sum(wins) / len(wins) if wins else None
    average_loss = abs(sum(losses) / len(losses)) if losses else None
    profit_loss_ratio = (
        (average_win / average_loss)
        if average_win is not None and average_loss not in (None, 0)
        else None
    )

    return RoundTripStats(
        closed_trade_count=closed_trade_count,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        average_win=average_win,
        average_loss=average_loss,
        gross_profit=sum(wins) if wins else 0.0,
        gross_loss=abs(sum(losses)) if losses else 0.0,
    )


def build_run_config(
    args: argparse.Namespace,
    history_range: HistoryRange,
    strategy_config_path: Path,
    strategy_config: dict[str, Any],
    instrument_meta: InstrumentMeta,
    strategy_setting: dict[str, Any],
    setting_overrides: dict[str, Any],
    absolute_slippage: float,
    output_dir: Path,
    data_check_summary: dict[str, Any] | None = None,
    setting_overrides_file_path: Path | None = None,
) -> dict[str, Any]:
    """Build final run_config.json payload."""

    return {
        "warning": WARNING_TEXT,
        "strategy_config_path": strategy_config_path,
        "instrument_config_path": INSTRUMENT_CONFIG_PATH,
        "setting_overrides": setting_overrides,
        "setting_overrides_file_path": setting_overrides_file_path,
        "output_dir": output_dir,
        "vt_symbol": args.vt_symbol,
        "start": args.start,
        "end": args.end,
        "data_timezone": history_range.timezone_name,
        "data_start": history_range.start,
        "data_end_exclusive": history_range.end_exclusive,
        "data_end_display": history_range.end_display,
        "capital": args.capital,
        "rate": args.rate,
        "slippage_mode": args.slippage_mode,
        "slippage_input": args.slippage,
        "absolute_slippage": absolute_slippage,
        "mode": args.mode,
        "engine_interval": "1m",
        "engine_mode": "BAR",
        "instrument_meta": instrument_meta,
        "strategy_class_name": strategy_config.get("class_name"),
        "strategy_name": strategy_config.get("strategy_name"),
        "strategy_setting": strategy_setting,
        "data_check_summary": data_check_summary,
    }


def calculate_statistics_safely(
    engine: Any,
    daily_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[dict[str, Any], str | None]:
    """Call vn.py statistics with exception-safe fallback."""

    try:
        statistics = engine.calculate_statistics(df=daily_df.copy(), output=False)
    except Exception as exc:
        log_event(
            logger,
            logging.WARNING,
            "backtest.statistics_exception",
            "calculate_statistics raised unexpectedly, diagnostics export will continue",
            error=repr(exc),
        )
        return {}, repr(exc)

    if not isinstance(statistics, dict):
        invalid_reason = f"calculate_statistics returned non-dict: {type(statistics).__name__}"
        log_event(
            logger,
            logging.WARNING,
            "backtest.statistics_invalid_type",
            "calculate_statistics returned unexpected type",
            invalid_reason=invalid_reason,
        )
        return {}, invalid_reason

    return dict(statistics), None


def build_stats_payload(
    statistics: dict[str, Any],
    diagnostics: dict[str, Any],
    round_trip_stats: RoundTripStats,
    engine_trade_count: int,
    order_count: int,
    statistics_error: str | None = None,
) -> dict[str, Any]:
    """Combine engine statistics, manual diagnostics, and round-trip trade stats."""

    bankrupt = bool(diagnostics.get("bankrupt", False))
    statistics_valid = bool(statistics) and not bankrupt and not statistics_error

    if bankrupt:
        invalid_reason = "balance <= 0 during backtest"
    elif statistics_error:
        invalid_reason = statistics_error
    elif not statistics:
        invalid_reason = "calculate_statistics returned empty result"
    else:
        invalid_reason = None

    stats_payload: dict[str, Any] = dict(statistics)
    stats_payload.update(
        {
            "warning": WARNING_TEXT,
            "bankrupt": bankrupt,
            "statistics_valid": statistics_valid,
            "invalid_reason": invalid_reason,
            "engine_trade_count": int(engine_trade_count),
            "order_count": int(order_count),
            "closed_round_trip_count": int(round_trip_stats.closed_trade_count),
            "statistics_total_trade_count": statistics.get("total_trade_count"),
            "win_rate": round_trip_stats.win_rate,
            "profit_loss_ratio": round_trip_stats.profit_loss_ratio,
            "average_win": round_trip_stats.average_win,
            "average_loss": round_trip_stats.average_loss,
            "gross_profit": round_trip_stats.gross_profit,
            "gross_loss": round_trip_stats.gross_loss,
            "first_bankrupt_date": diagnostics.get("first_bankrupt_date"),
            "final_balance": diagnostics.get("final_balance"),
            "min_balance": diagnostics.get("min_balance"),
            "max_balance": diagnostics.get("max_balance"),
        }
    )
    return stats_payload


def export_chart_html(engine: Any, daily_df: pd.DataFrame, output_path: Path) -> None:
    """Export plotly chart HTML with a placeholder fallback."""

    figure = engine.show_chart(daily_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if figure is not None:
        figure.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    output_path.write_text(
        "<html><body><p>No chart available.</p></body></html>",
        encoding="utf-8",
    )


def run_data_preflight(
    args: argparse.Namespace,
    history_range: HistoryRange,
    logger: logging.Logger,
) -> tuple[dict[str, Any], HistoryRange]:
    """Verify sqlite coverage before backtest load_data starts."""

    if args.skip_data_check:
        summary = {
            "skipped": True,
            "strict": bool(args.data_check_strict),
            "reason": "--skip-data-check",
            "timezone": history_range.timezone_name,
            "start": history_range.start,
            "end_exclusive": history_range.end_exclusive,
            "end_display": history_range.end_display,
        }
        log_event(
            logger,
            logging.WARNING,
            "backtest.data_check_skipped",
            "Skipping sqlite history preflight before backtest",
            reason="--skip-data-check",
        )
        return summary, history_range

    result = verify_history_range(
        vt_symbol=args.vt_symbol,
        interval_value="1m",
        start_arg=args.start,
        end_arg=args.end,
        timezone_name=args.timezone,
        history_range=history_range,
    )
    summary = build_data_check_payload(result)
    summary["skipped"] = False
    summary["strict"] = bool(args.data_check_strict)
    print_json_block("Backtest data check:", summary)

    if result.is_empty:
        raise BacktestError(
            "回测前数据库预检查失败：本地数据库没有所需历史数据。"
            f" 修复命令: {result.repair_command}"
        )

    if args.data_check_strict and not result.is_complete:
        raise BacktestError(
            "回测前数据库预检查失败：本地数据库历史数据存在缺口。"
            f" 修复命令: {result.repair_command}"
        )

    if not result.is_complete:
        log_event(
            logger,
            logging.WARNING,
            "backtest.data_check_partial",
            "Backtest data check found gaps, but strict mode is disabled",
            missing_count=result.coverage.missing_count,
            gap_count=result.coverage.gap_count,
            repair_command=result.repair_command,
        )
    else:
        log_event(
            logger,
            logging.INFO,
            "backtest.data_check_complete",
            "Backtest data check passed",
            total_count=result.coverage.total_count,
        )

    return summary, result.history_range


def to_engine_datetime(value: datetime) -> datetime:
    """Convert an aware history boundary into the naive sqlite timezone used by BacktestingEngine."""

    db_tz = get_database_timezone()
    return value.astimezone(db_tz).replace(tzinfo=None)


def main() -> int:
    """Run the local database backtest and export reports."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("backtest_okx_mhf", verbose=args.verbose)
    strategy_config_path: Path = resolve_path_arg(args.strategy_config, STRATEGY_CONFIG_PATH)
    instrument_config_path: Path = INSTRUMENT_CONFIG_PATH

    try:
        from vnpy.trader.constant import Interval
        from vnpy_ctastrategy.backtesting import BacktestingEngine, BacktestingMode

        _interval, interval_delta = parse_interval("1m")
        try:
            history_range = parse_history_range(
                start_arg=args.start,
                end_arg=args.end,
                interval_delta=interval_delta,
                timezone_name=args.timezone,
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        engine_start = to_engine_datetime(history_range.start)
        engine_end = to_engine_datetime(history_range.end_display)

        strategy_config = load_json_file(strategy_config_path)
        instrument_meta = load_instrument_meta(instrument_config_path)
        validate_runtime_config(args.vt_symbol, instrument_meta, history_range.start, history_range.end_display)
        setting_overrides, setting_overrides_file_path = load_setting_overrides(
            args.setting_overrides_file,
            args.setting_overrides,
        )

        raw_setting = strategy_config.get("setting")
        if not isinstance(raw_setting, dict):
            raise ConfigurationError(f"策略配置缺少 setting 对象: {strategy_config_path}")

        strategy_setting = fill_strategy_setting(
            raw_setting,
            instrument_meta,
            args.capital,
            logger,
            setting_overrides=setting_overrides,
        )
        strategy_class_name = str(strategy_config.get("class_name", "")).strip()
        if not strategy_class_name:
            raise ConfigurationError(f"策略配置缺少 class_name: {strategy_config_path}")

        strategy_class = resolve_strategy_class(strategy_class_name)

        absolute_slippage = resolve_slippage(
            pricetick=instrument_meta.pricetick,
            slippage_mode=args.slippage_mode,
            slippage=args.slippage,
        )

        output_dir = resolve_output_dir(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        warning_path = output_dir / "warning.txt"
        warning_path.write_text(WARNING_TEXT + "\n", encoding="utf-8")

        configure_sqlite_settings(logger)
        data_check_summary, history_range = run_data_preflight(args, history_range, logger)

        run_config = build_run_config(
            args=args,
            history_range=history_range,
            strategy_config_path=strategy_config_path,
            strategy_config=strategy_config,
            instrument_meta=instrument_meta,
            strategy_setting=strategy_setting,
            setting_overrides=setting_overrides,
            absolute_slippage=absolute_slippage,
            output_dir=output_dir,
            data_check_summary=data_check_summary,
            setting_overrides_file_path=setting_overrides_file_path,
        )
        write_json_file(output_dir / "run_config.json", run_config)

        engine = BacktestingEngine()
        engine.output = make_engine_output(logger)

        log_event(
            logger,
            logging.INFO,
            "backtest.start",
            "Starting backtest",
            vt_symbol=args.vt_symbol,
            timezone=history_range.timezone_name,
            start=history_range.start,
            end_exclusive=history_range.end_exclusive,
            end_display=history_range.end_display,
            engine_start=engine_start,
            engine_end=engine_end,
            capital=args.capital,
            rate=args.rate,
            absolute_slippage=absolute_slippage,
            size=instrument_meta.size,
            pricetick=instrument_meta.pricetick,
            min_volume=instrument_meta.min_volume,
        )

        engine.set_parameters(
            vt_symbol=args.vt_symbol,
            interval=Interval.MINUTE,
            start=engine_start,
            rate=args.rate,
            slippage=absolute_slippage,
            size=instrument_meta.size,
            pricetick=instrument_meta.pricetick,
            capital=args.capital,
            end=engine_end,
            mode=BacktestingMode.BAR,
        )
        engine.add_strategy(strategy_class, strategy_setting)
        engine.load_data()

        if not engine.history_data:
            raise BacktestError(
                f"本地数据库中没有可用的 1m 数据: vt_symbol={args.vt_symbol}, "
                f"start={history_range.start}, end_display={history_range.end_display}"
            )

        engine.run_backtesting()
        daily_df = engine.calculate_result()
        if daily_df is None or daily_df.empty:
            raise BacktestError("calculate_result 返回空结果，无法生成报告")

        prepared_daily_df, _prepared_missing_columns = prepare_daily_pnl_dataframe(daily_df, args.capital)
        diagnostics = analyze_daily_pnl(prepared_daily_df, args.capital)
        statistics, statistics_error = calculate_statistics_safely(engine, prepared_daily_df, logger)

        trades = engine.get_all_trades()
        orders = engine.get_all_orders()
        round_trip_stats = compute_round_trip_stats(trades, instrument_meta.size)
        engine_trade_count = len(trades)
        order_count = len(orders)

        if diagnostics.get("bankrupt") and not args.allow_bankrupt_report:
            raise BacktestError(
                "检测到回测资金小于等于 0，且 --allow-bankrupt-report 已关闭，停止报告导出。"
            )

        trade_records = [trade_to_record(trade) for trade in trades]
        order_records = [order_to_record(order) for order in orders]

        stats_payload = build_stats_payload(
            statistics=statistics,
            diagnostics=diagnostics,
            round_trip_stats=round_trip_stats,
            engine_trade_count=engine_trade_count,
            order_count=order_count,
            statistics_error=statistics_error,
        )

        diagnostics["statistics_error"] = statistics_error
        diagnostics["statistics_total_trade_count"] = statistics.get("total_trade_count")
        diagnostics["engine_trade_count"] = engine_trade_count
        diagnostics["order_count"] = order_count
        diagnostics["closed_round_trip_count"] = round_trip_stats.closed_trade_count

        daily_output_df = prepared_daily_df.reset_index()
        chart_path = output_dir / "chart.html"
        export_errors: dict[str, str] = {}

        exports: list[tuple[str, Callable[[], None]]] = [
            ("daily_pnl.csv", lambda: export_dataframe_csv(daily_output_df, output_dir / "daily_pnl.csv")),
            (
                "trades.csv",
                lambda: export_records_csv(
                    trade_records,
                    output_dir / "trades.csv",
                    columns=[
                        "datetime",
                        "vt_tradeid",
                        "vt_orderid",
                        "symbol",
                        "exchange",
                        "direction",
                        "offset",
                        "price",
                        "volume",
                        "gateway_name",
                        "tradeid",
                        "orderid",
                    ],
                ),
            ),
            (
                "orders.csv",
                lambda: export_records_csv(
                    order_records,
                    output_dir / "orders.csv",
                    columns=[
                        "datetime",
                        "vt_orderid",
                        "symbol",
                        "exchange",
                        "type",
                        "direction",
                        "offset",
                        "price",
                        "volume",
                        "traded",
                        "status",
                        "gateway_name",
                        "orderid",
                        "reference",
                    ],
                ),
            ),
            ("chart.html", lambda: export_chart_html(engine, prepared_daily_df, chart_path)),
            ("stats.json", lambda: write_json_file(output_dir / "stats.json", stats_payload)),
            ("diagnostics.json", lambda: write_json_file(output_dir / "diagnostics.json", diagnostics)),
        ]

        for artifact_name, exporter in exports:
            try:
                exporter()
            except Exception as exc:
                export_errors[artifact_name] = repr(exc)
                log_event(
                    logger,
                    logging.WARNING,
                    "backtest.report_export_failed",
                    "Failed to export backtest artifact",
                    artifact_name=artifact_name,
                    output_dir=output_dir,
                    error=repr(exc),
                )

        run_config["generated_files"] = {
            "warning_txt": warning_path,
            "run_config_json": output_dir / "run_config.json",
            "stats_json": output_dir / "stats.json",
            "diagnostics_json": output_dir / "diagnostics.json",
            "daily_pnl_csv": output_dir / "daily_pnl.csv",
            "trades_csv": output_dir / "trades.csv",
            "orders_csv": output_dir / "orders.csv",
            "chart_html": chart_path,
        }
        run_config["export_errors"] = export_errors
        write_json_file(output_dir / "run_config.json", run_config)

        summary = {
            "warning": WARNING_TEXT,
            "output_dir": output_dir,
            "bankrupt": stats_payload.get("bankrupt"),
            "statistics_valid": stats_payload.get("statistics_valid"),
            "invalid_reason": stats_payload.get("invalid_reason"),
            "first_bankrupt_date": stats_payload.get("first_bankrupt_date"),
            "min_balance": stats_payload.get("min_balance"),
            "final_balance": stats_payload.get("final_balance"),
            "engine_trade_count": stats_payload.get("engine_trade_count"),
            "closed_round_trip_count": stats_payload.get("closed_round_trip_count"),
            "order_count": stats_payload.get("order_count"),
            "total_net_pnl": stats_payload.get("total_net_pnl"),
            "annual_return": stats_payload.get("annual_return"),
            "max_drawdown": stats_payload.get("max_drawdown"),
            "max_ddpercent": stats_payload.get("max_ddpercent"),
            "sharpe_ratio": stats_payload.get("sharpe_ratio"),
            "return_drawdown_ratio": stats_payload.get("return_drawdown_ratio"),
            "win_rate": stats_payload.get("win_rate"),
            "profit_loss_ratio": stats_payload.get("profit_loss_ratio"),
            "statistics_total_trade_count": stats_payload.get("statistics_total_trade_count"),
            "export_errors": export_errors,
        }

        print_json_block("Backtest summary:", summary)
        return 0
    except ConfigurationError as exc:
        log_event(
            logger,
            logging.ERROR,
            "backtest.config_error",
            str(exc),
            strategy_config_path=strategy_config_path,
            instrument_config_path=instrument_config_path,
        )
        return 1
    except BacktestError as exc:
        log_event(
            logger,
            logging.ERROR,
            "backtest.execution_error",
            str(exc),
        )
        return 2
    except Exception:
        logger.exception(
            "Unexpected error during backtest",
            extra={"event": "backtest.unexpected_error"},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
