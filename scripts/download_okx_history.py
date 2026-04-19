#!/usr/bin/env python3
"""Download OKX bar history and save it into the local vn.py database."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING, Any

from dotenv import dotenv_values

from common_runtime import (
    PROJECT_ROOT,
    ensure_headless_runtime,
    log_event,
    normalize_text,
    print_json_block,
    setup_logging,
    to_jsonable,
)

if TYPE_CHECKING:
    from vnpy.event import Event
    from vnpy.trader.constant import Exchange, Interval
    from vnpy.trader.object import BarData, ContractData, LogData


ENV_FILE: Path = PROJECT_ROOT / ".env"
DEFAULT_GATEWAY_NAME: str = "OKX"
DEFAULT_VT_SYMBOL: str = "BTCUSDT_SWAP_OKX.GLOBAL"
DEFAULT_INTERVAL: str = "1m"
DEFAULT_DATABASE_NAME: str = "sqlite"
DEFAULT_DATABASE_FILE: str = "database.db"


class ConfigurationError(Exception):
    """Raised when local runtime or CLI configuration is invalid."""


class ConnectionError(Exception):
    """Raised when OKX connection bootstrap fails."""


class HistoryDownloadError(Exception):
    """Raised when one or more history download steps fail."""


class DatabaseSaveError(Exception):
    """Raised when bar data cannot be saved into vn.py database."""


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Connection settings read from project .env."""

    api_key: str
    secret_key: str
    passphrase: str
    server: str
    proxy_host: str
    proxy_port: int


@dataclass(frozen=True, slots=True)
class GatewayFieldMap:
    """Resolved OKX gateway field names for the local vnpy_okx version."""

    api_key: str
    secret_key: str
    passphrase: str
    server: str
    proxy_host: str | None
    proxy_port: str | None


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """One history query chunk."""

    index: int
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """One history chunk result summary."""

    index: int
    start: datetime
    end: datetime
    count: int
    first_dt: datetime | None
    last_dt: datetime | None


@dataclass(frozen=True, slots=True)
class GapInfo:
    """A detected gap in bar history."""

    previous_dt: datetime
    current_dt: datetime
    missing_bars: int
    delta_seconds: float


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    """Final download summary for console output."""

    vt_symbol: str
    interval: str
    requested_start: datetime
    requested_end: datetime
    total_raw_count: int
    total_unique_count: int
    duplicate_count: int
    first_dt: datetime
    last_dt: datetime
    gap_count: int
    chunk_results: list[ChunkResult]
    csv_path: Path | None
    instrument_path: Path


class GatewayObserver:
    """Capture gateway logs and target contract data during bootstrap."""

    def __init__(self, logger: logging.Logger, vt_symbol: str, verbose: bool) -> None:
        self.logger: logging.Logger = logger
        self.vt_symbol: str = vt_symbol
        self.verbose: bool = verbose
        self.contract: ContractData | None = None
        self.contract_ready: ThreadingEvent = ThreadingEvent()
        self.private_login_success: bool = False
        self.private_login_failed: bool = False
        self.recent_logs: deque[dict[str, Any]] = deque(maxlen=300)

    def on_log(self, event: "Event") -> None:
        """Capture vn.py log events for diagnostics."""

        log_data: LogData = event.data
        item: dict[str, Any] = {
            "gateway_name": getattr(log_data, "gateway_name", ""),
            "msg": getattr(log_data, "msg", ""),
            "time": getattr(log_data, "time", None),
        }
        self.recent_logs.append(item)

        message_lower: str = item["msg"].lower()
        if "private api login successful" in message_lower:
            self.private_login_success = True
        elif "private api login failed" in message_lower:
            self.private_login_failed = True

        if self.verbose:
            log_event(
                self.logger,
                logging.DEBUG,
                "okx.gateway_log",
                "Captured vn.py gateway log",
                gateway_name=item["gateway_name"],
                log_message=item["msg"],
            )

    def on_contract(self, event: "Event") -> None:
        """Capture target contract once it is pushed by the gateway."""

        contract: ContractData = event.data
        if contract.vt_symbol != self.vt_symbol:
            return

        self.contract = contract
        self.contract_ready.set()
        log_event(
            self.logger,
            logging.INFO,
            "okx.contract_ready",
            "Target contract metadata received",
            vt_symbol=contract.vt_symbol,
            contract_name=contract.name,
            gateway_name=contract.gateway_name,
            history_data=getattr(contract, "history_data", None),
        )

    def recent_log_messages(self, limit: int = 30) -> list[dict[str, Any]]:
        """Return latest captured gateway logs."""

        return list(self.recent_logs)[-limit:]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Download OKX history data and save it into vn.py sqlite database."
    )
    parser.add_argument(
        "--vt-symbol",
        default=DEFAULT_VT_SYMBOL,
        help=f"Target vt_symbol. Default: {DEFAULT_VT_SYMBOL}.",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=("1m", "1h", "d", "w"),
        help="History bar interval. Default: 1m.",
    )
    parser.add_argument(
        "--start",
        default="2025-01-01",
        help="Inclusive start datetime in ISO format. Date-only means 00:00:00 UTC.",
    )
    parser.add_argument(
        "--end",
        default="2026-03-31",
        help="Inclusive end datetime in ISO format. Date-only means 23:59:59.999999 UTC.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=5,
        help="Chunk size in days. Default: 5.",
    )
    parser.add_argument(
        "--server",
        choices=("DEMO", "REAL"),
        help="Override OKX_SERVER from .env.",
    )
    parser.add_argument(
        "--csv-copy",
        action="store_true",
        help="Export a CSV copy into data/raw/ after database save.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Connection bootstrap timeout in seconds. Default: 30.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def parse_datetime_arg(value: str, is_end: bool) -> datetime:
    """Parse CLI datetime input and normalize it to timezone-aware UTC."""

    value = value.strip()
    if not value:
        raise ConfigurationError("时间参数不能为空")

    if len(value) == 10:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigurationError(f"无法解析日期参数: {value}") from exc
        naive_dt = datetime.combine(
            parsed_date,
            dt_time.max if is_end else dt_time.min,
        )
    else:
        try:
            naive_dt = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ConfigurationError(f"无法解析时间参数: {value}") from exc

    if naive_dt.tzinfo is None:
        return naive_dt.replace(tzinfo=timezone.utc)
    return naive_dt.astimezone(timezone.utc)


def configure_sqlite_settings(logger: logging.Logger) -> None:
    """Force vn.py runtime database settings to local sqlite database."""

    from vnpy.trader.setting import SETTINGS

    SETTINGS["database.name"] = DEFAULT_DATABASE_NAME
    SETTINGS["database.database"] = DEFAULT_DATABASE_FILE

    log_event(
        logger,
        logging.INFO,
        "db.settings_configured",
        "Configured vn.py database settings for sqlite",
        database_name=SETTINGS["database.name"],
        database_database=SETTINGS["database.database"],
    )


def read_env_config(server_override: str | None) -> EnvConfig:
    """Read OKX credentials and proxy settings from project .env."""

    if not ENV_FILE.exists():
        raise ConfigurationError(
            f".env not found at {ENV_FILE}. Create it first with: cp .env.example .env"
        )

    raw_values: dict[str, str | None] = dotenv_values(ENV_FILE)
    values: dict[str, str] = {
        key: (value.strip() if isinstance(value, str) else "")
        for key, value in raw_values.items()
    }

    required_fields: tuple[str, ...] = (
        "OKX_API_KEY",
        "OKX_SECRET_KEY",
        "OKX_PASSPHRASE",
        "OKX_SERVER",
    )
    missing_fields: list[str] = [field for field in required_fields if not values.get(field)]
    if server_override:
        missing_fields = [field for field in missing_fields if field != "OKX_SERVER"]

    if missing_fields:
        raise ConfigurationError(f".env 缺字段或为空: {', '.join(missing_fields)}")

    proxy_port_raw: str = values.get("OKX_PROXY_PORT", "0") or "0"
    try:
        proxy_port = int(proxy_port_raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"OKX_PROXY_PORT 必须是整数，当前值为: {proxy_port_raw}"
        ) from exc

    proxy_host: str = values.get("OKX_PROXY_HOST", "")
    if proxy_host and proxy_port <= 0:
        raise ConfigurationError("代理配置错误：OKX_PROXY_HOST 已设置，但 OKX_PROXY_PORT <= 0")
    if proxy_port > 0 and not proxy_host:
        raise ConfigurationError("代理配置错误：OKX_PROXY_PORT 已设置，但 OKX_PROXY_HOST 为空")

    server_value: str = (server_override or values.get("OKX_SERVER", "")).upper()
    return EnvConfig(
        api_key=values["OKX_API_KEY"],
        secret_key=values["OKX_SECRET_KEY"],
        passphrase=values["OKX_PASSPHRASE"],
        server=server_value,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
    )


def coerce_default_value(value: Any) -> Any:
    """Convert gateway default-setting values into connect-ready values."""

    if isinstance(value, list):
        return value[0] if value else ""
    return value


def find_setting_key(default_setting: dict[str, Any], *tokens: str) -> str | None:
    """Find a gateway field key by semantic tokens instead of hard-coding labels."""

    for key in default_setting:
        normalized_key = normalize_text(key)
        if all(normalize_text(token) in normalized_key for token in tokens):
            return key
    return None


def resolve_gateway_field_map(default_setting: dict[str, Any]) -> GatewayFieldMap:
    """Resolve the local OKX gateway field names dynamically."""

    api_key = find_setting_key(default_setting, "api", "key")
    secret_key = find_setting_key(default_setting, "secret", "key")
    passphrase = find_setting_key(default_setting, "passphrase")
    server = find_setting_key(default_setting, "server")
    proxy_host = find_setting_key(default_setting, "proxy", "host")
    proxy_port = find_setting_key(default_setting, "proxy", "port")

    missing_fields: list[str] = []
    if not api_key:
        missing_fields.append("api_key")
    if not secret_key:
        missing_fields.append("secret_key")
    if not passphrase:
        missing_fields.append("passphrase")
    if not server:
        missing_fields.append("server")

    if missing_fields:
        raise ConfigurationError(
            f"无法从 OKX default_setting 动态解析必需字段: {', '.join(missing_fields)}"
        )

    return GatewayFieldMap(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        server=server,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
    )


def validate_server_option(
    default_setting: dict[str, Any],
    field_map: GatewayFieldMap,
    server: str,
) -> None:
    """Validate server override against local vnpy_okx options."""

    server_value = default_setting.get(field_map.server)
    if isinstance(server_value, list) and server not in server_value:
        raise ConfigurationError(
            f"Server 取值无效: {server}，当前本地 OKX 可选项为: {', '.join(server_value)}"
        )


def build_connection_setting(
    default_setting: dict[str, Any],
    env_config: EnvConfig,
    field_map: GatewayFieldMap,
) -> dict[str, Any]:
    """Assemble connect setting dict from dynamic gateway fields and .env."""

    validate_server_option(default_setting, field_map, env_config.server)

    setting: dict[str, Any] = {
        key: coerce_default_value(value)
        for key, value in default_setting.items()
    }
    setting[field_map.api_key] = env_config.api_key
    setting[field_map.secret_key] = env_config.secret_key
    setting[field_map.passphrase] = env_config.passphrase
    setting[field_map.server] = env_config.server

    if field_map.proxy_host is not None:
        setting[field_map.proxy_host] = env_config.proxy_host
    if field_map.proxy_port is not None:
        setting[field_map.proxy_port] = env_config.proxy_port

    return setting


def sanitize_setting(setting: dict[str, Any]) -> dict[str, Any]:
    """Mask secret values before logging connection settings."""

    sanitized: dict[str, Any] = {}
    for key, value in setting.items():
        normalized_key = normalize_text(key)
        if isinstance(value, str) and any(
            token in normalized_key for token in ("apikey", "secret", "passphrase")
        ):
            visible_suffix = value[-4:] if value else ""
            sanitized[key] = f"{'*' * max(len(value) - 4, 0)}{visible_suffix}"
        else:
            sanitized[key] = to_jsonable(value)
    return sanitized


def parse_interval(interval_value: str) -> tuple["Interval", timedelta]:
    """Map CLI interval text to vn.py Interval and expected gap delta."""

    from vnpy.trader.constant import Interval

    mapping: dict[str, tuple[Interval, timedelta]] = {
        "1m": (Interval.MINUTE, timedelta(minutes=1)),
        "1h": (Interval.HOUR, timedelta(hours=1)),
        "d": (Interval.DAILY, timedelta(days=1)),
        "w": (Interval.WEEKLY, timedelta(weeks=1)),
    }

    try:
        return mapping[interval_value]
    except KeyError as exc:
        raise ConfigurationError(f"不支持的 interval: {interval_value}") from exc


def infer_connection_failure(observer: GatewayObserver, vt_symbol: str) -> str:
    """Infer clearer connection failure reason from recent vn.py gateway logs."""

    combined_logs: str = "\n".join(item["msg"] for item in observer.recent_logs)
    logs_lower: str = combined_logs.lower()

    if "proxy" in logs_lower and any(
        token in logs_lower for token in ("refused", "error", "failed", "timeout")
    ):
        return "代理配置错误，无法通过代理建立到 OKX 的连接。"

    if any(
        token in logs_lower
        for token in (
            "private api login failed",
            "login failed",
            "api key",
            "passphrase",
            "status code: 60009",
            "status code: 60024",
            "status code: 60013",
        )
    ):
        return "API Key / Secret / Passphrase 不正确，或 DEMO / REAL 不匹配。"

    if any(
        token in logs_lower
        for token in (
            "name resolution",
            "network is unreachable",
            "max retries exceeded",
            "timed out",
            "ssl",
            "connection aborted",
            "exception catched by rest api",
            "exception catched by public api",
            "exception catched by private api",
        )
    ):
        return "网络问题，或代理配置错误，导致无法连接 OKX。"

    if observer.contract is None:
        return f"在超时时间内没有拿到目标合约元数据: {vt_symbol}"

    return f"已拿到目标合约元数据，但连接状态不完整，请检查 DEMO/REAL、网络和 API 权限: {vt_symbol}"


def wait_for_contract(
    main_engine: Any,
    observer: GatewayObserver,
    vt_symbol: str,
    timeout: float,
    logger: logging.Logger,
) -> ContractData:
    """Wait until target contract appears, while watching for obvious login failures."""

    deadline: float = time.monotonic() + timeout
    next_progress_log: float = time.monotonic() + 5.0
    contract_detected_at: float | None = None
    contract_grace_seconds: float = 3.0

    while time.monotonic() < deadline:
        contract = observer.contract or main_engine.get_contract(vt_symbol)
        if contract is not None:
            observer.contract = contract
            observer.contract_ready.set()
            if contract_detected_at is None:
                contract_detected_at = time.monotonic()

        if observer.private_login_failed:
            raise ConnectionError(
                "API Key / Secret / Passphrase 不正确，或 DEMO / REAL 不匹配，Private API 登录失败。"
            )

        if observer.contract is not None and observer.private_login_success:
            return observer.contract

        if observer.contract is not None and contract_detected_at is not None:
            if time.monotonic() - contract_detected_at >= contract_grace_seconds:
                log_event(
                    logger,
                    logging.WARNING,
                    "okx.contract_only_ready",
                    "Target contract is ready, but private login success was not observed within grace period; continue with public history query",
                    vt_symbol=vt_symbol,
                )
                return observer.contract

        now = time.monotonic()
        if now >= next_progress_log:
            log_event(
                logger,
                logging.INFO,
                "okx.waiting_contract",
                "Waiting for OKX contract metadata",
                vt_symbol=vt_symbol,
                contract_ready=observer.contract_ready.is_set(),
                private_login_success=observer.private_login_success,
            )
            next_progress_log = now + 5.0

        time.sleep(0.5)

    raise ConnectionError(infer_connection_failure(observer, vt_symbol))


def build_history_chunks(
    start: datetime,
    end: datetime,
    interval_delta: timedelta,
    chunk_days: int,
) -> list[ChunkPlan]:
    """Build inclusive chunk ranges for history download."""

    if end < start:
        raise ConfigurationError(f"结束时间早于开始时间: start={start}, end={end}")
    if chunk_days <= 0:
        raise ConfigurationError(f"chunk-days 必须大于 0，当前值为: {chunk_days}")
    if interval_delta <= timedelta(0):
        raise ConfigurationError("interval delta 必须大于 0")

    chunk_span: timedelta = timedelta(days=chunk_days)
    cursor: datetime = start
    chunks: list[ChunkPlan] = []
    index: int = 1

    while cursor <= end:
        chunk_end: datetime = min(cursor + chunk_span - interval_delta, end)
        chunks.append(ChunkPlan(index=index, start=cursor, end=chunk_end))
        cursor = chunk_end + interval_delta
        index += 1

    return chunks


def query_history_by_chunks(
    main_engine: Any,
    gateway_name: str,
    symbol: str,
    exchange: "Exchange",
    interval: "Interval",
    chunks: list[ChunkPlan],
    logger: logging.Logger,
) -> tuple[list[BarData], list[ChunkResult], int]:
    """Download history chunk by chunk, then deduplicate and sort by datetime."""

    from vnpy.trader.object import HistoryRequest

    bars_by_dt: dict[datetime, BarData] = {}
    chunk_results: list[ChunkResult] = []
    duplicate_count: int = 0

    for chunk in chunks:
        request = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            start=chunk.start,
            end=chunk.end,
            interval=interval,
        )

        try:
            bars: list[BarData] = main_engine.query_history(request, gateway_name)
        except Exception as exc:
            raise HistoryDownloadError(
                f"第 {chunk.index} 个 chunk 查询失败: {chunk.start.isoformat()} -> {chunk.end.isoformat()} | {exc!r}"
            ) from exc

        bars.sort(key=lambda bar: bar.datetime)
        for bar in bars:
            if bar.datetime in bars_by_dt:
                duplicate_count += 1
            bars_by_dt[bar.datetime] = bar

        first_dt: datetime | None = bars[0].datetime if bars else None
        last_dt: datetime | None = bars[-1].datetime if bars else None
        result = ChunkResult(
            index=chunk.index,
            start=chunk.start,
            end=chunk.end,
            count=len(bars),
            first_dt=first_dt,
            last_dt=last_dt,
        )
        chunk_results.append(result)

        log_event(
            logger,
            logging.INFO,
            "history.chunk_done",
            "Downloaded one history chunk",
            chunk_index=chunk.index,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
            count=len(bars),
            first_dt=first_dt,
            last_dt=last_dt,
        )

    unique_bars: list[BarData] = [bars_by_dt[dt] for dt in sorted(bars_by_dt)]
    return unique_bars, chunk_results, duplicate_count


def detect_time_gaps(bars: list[BarData], interval_delta: timedelta) -> list[GapInfo]:
    """Detect obvious time gaps in sorted bar history."""

    if len(bars) < 2:
        return []

    gaps: list[GapInfo] = []
    expected_seconds: float = interval_delta.total_seconds()
    for previous_bar, current_bar in zip(bars[:-1], bars[1:]):
        delta_seconds: float = (current_bar.datetime - previous_bar.datetime).total_seconds()
        if delta_seconds > expected_seconds:
            missing_bars = int(round(delta_seconds / expected_seconds)) - 1
            gaps.append(
                GapInfo(
                    previous_dt=previous_bar.datetime,
                    current_dt=current_bar.datetime,
                    missing_bars=max(missing_bars, 1),
                    delta_seconds=delta_seconds,
                )
            )

    return gaps


def export_csv_copy(
    vt_symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    bars: list[BarData],
) -> Path:
    """Export a CSV copy of downloaded bar data into data/raw."""

    symbol_part: str = vt_symbol.split(".", maxsplit=1)[0].lower()
    filename: str = (
        f"{symbol_part}_{interval}_"
        f"{start.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{end.strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    output_path: Path = PROJECT_ROOT / "data" / "raw" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "datetime",
                "symbol",
                "exchange",
                "interval",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "turnover",
                "open_interest",
                "gateway_name",
            ],
        )
        writer.writeheader()

        for bar in bars:
            writer.writerow(
                {
                    "datetime": bar.datetime.isoformat(),
                    "symbol": bar.symbol,
                    "exchange": bar.exchange.value,
                    "interval": bar.interval.value,
                    "open_price": bar.open_price,
                    "high_price": bar.high_price,
                    "low_price": bar.low_price,
                    "close_price": bar.close_price,
                    "volume": bar.volume,
                    "turnover": bar.turnover,
                    "open_interest": bar.open_interest,
                    "gateway_name": bar.gateway_name,
                }
            )

    return output_path


def extract_contract_payload(contract: ContractData) -> dict[str, Any]:
    """Extract portable contract metadata for local config persistence."""

    field_names: tuple[str, ...] = (
        "vt_symbol",
        "symbol",
        "exchange",
        "name",
        "size",
        "pricetick",
        "min_volume",
        "gateway_name",
        "history_data",
    )

    payload: dict[str, Any] = {}
    for field_name in field_names:
        if hasattr(contract, field_name):
            payload[field_name] = to_jsonable(getattr(contract, field_name))
    return payload


def build_instrument_output_path(vt_symbol: str) -> Path:
    """Build config/instruments output path from vt_symbol."""

    symbol_part, separator, _exchange_part = vt_symbol.partition(".")
    if not separator or not symbol_part:
        raise ConfigurationError(f"vt_symbol 格式无效: {vt_symbol}")
    return PROJECT_ROOT / "config" / "instruments" / f"{symbol_part.lower()}.json"


def save_contract_payload(payload: dict[str, Any], output_path: Path) -> None:
    """Write contract metadata JSON to config/instruments."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_bars_to_database(bars: list[BarData], logger: logging.Logger) -> None:
    """Save deduplicated bars into vn.py local database."""

    if not bars:
        raise DatabaseSaveError("没有可写入数据库的 bar 数据")

    try:
        from vnpy.trader.database import get_database
        from vnpy.trader.setting import SETTINGS
    except Exception as exc:
        raise DatabaseSaveError(
            f"无法导入 vn.py 数据库模块，请检查 vnpy_sqlite 安装和环境: {exc!r}"
        ) from exc

    try:
        database = get_database()
    except Exception as exc:
        raise DatabaseSaveError(
            "创建数据库实例失败。请检查数据库驱动或 SETTINGS 配置，不是策略问题。"
            f" 当前配置: database.name={SETTINGS.get('database.name')}, "
            f"database.database={SETTINGS.get('database.database')} | {exc!r}"
        ) from exc

    try:
        result: bool = database.save_bar_data(bars, stream=False)
    except Exception as exc:
        raise DatabaseSaveError(
            "save_bar_data 执行异常。优先检查数据库驱动和 SETTINGS 配置，不是策略问题。"
            f" 当前数据库实例: {type(database).__name__} | {exc!r}"
        ) from exc

    if not result:
        raise DatabaseSaveError(
            "save_bar_data 返回 False。优先检查数据库驱动和 SETTINGS 配置，不是策略问题。"
        )

    log_event(
        logger,
        logging.INFO,
        "db.save_success",
        "Saved bars into vn.py database",
        database_type=type(database).__name__,
        count=len(bars),
        first_dt=bars[0].datetime,
        last_dt=bars[-1].datetime,
    )


def build_final_summary(
    vt_symbol: str,
    interval: str,
    requested_start: datetime,
    requested_end: datetime,
    bars: list[BarData],
    raw_total_count: int,
    duplicate_count: int,
    gaps: list[GapInfo],
    chunk_results: list[ChunkResult],
    csv_path: Path | None,
    instrument_path: Path,
) -> DownloadSummary:
    """Build final download summary object."""

    return DownloadSummary(
        vt_symbol=vt_symbol,
        interval=interval,
        requested_start=requested_start,
        requested_end=requested_end,
        total_raw_count=raw_total_count,
        total_unique_count=len(bars),
        duplicate_count=duplicate_count,
        first_dt=bars[0].datetime,
        last_dt=bars[-1].datetime,
        gap_count=len(gaps),
        chunk_results=chunk_results,
        csv_path=csv_path,
        instrument_path=instrument_path,
    )


def main() -> int:
    """Run OKX history download."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("download_okx_history", verbose=args.verbose)

    main_engine = None
    observer: GatewayObserver | None = None

    try:
        from vnpy.event import EventEngine
        from vnpy.trader.engine import MainEngine
        from vnpy.trader.event import EVENT_CONTRACT, EVENT_LOG
        from vnpy.trader.setting import SETTINGS
        from vnpy.trader.utility import extract_vt_symbol
        from vnpy_okx import OkxGateway

        start_dt: datetime = parse_datetime_arg(args.start, is_end=False)
        end_dt: datetime = parse_datetime_arg(args.end, is_end=True)
        interval, interval_delta = parse_interval(args.interval)

        configure_sqlite_settings(logger)

        env_config = read_env_config(args.server)
        chunks = build_history_chunks(
            start=start_dt,
            end=end_dt,
            interval_delta=interval_delta,
            chunk_days=args.chunk_days,
        )

        log_event(
            logger,
            logging.INFO,
            "history.download_start",
            "Starting OKX history download",
            vt_symbol=args.vt_symbol,
            interval=args.interval,
            start=start_dt,
            end=end_dt,
            chunk_days=args.chunk_days,
            chunk_count=len(chunks),
            timeout=args.timeout,
            server=env_config.server,
            database_name=SETTINGS["database.name"],
            database_database=SETTINGS["database.database"],
        )

        event_engine = EventEngine()
        main_engine = MainEngine(event_engine)
        main_engine.add_gateway(OkxGateway)

        default_setting = main_engine.get_default_setting(DEFAULT_GATEWAY_NAME)
        if default_setting is None:
            raise ConfigurationError("无法读取 OKX default_setting。请确认 OkxGateway 已正确安装。")

        field_map = resolve_gateway_field_map(default_setting)
        connect_setting = build_connection_setting(default_setting, env_config, field_map)

        observer = GatewayObserver(logger, args.vt_symbol, args.verbose)
        event_engine.register(EVENT_LOG, observer.on_log)
        event_engine.register(EVENT_CONTRACT, observer.on_contract)

        log_event(
            logger,
            logging.INFO,
            "okx.connect",
            "Connecting to OKX gateway for history download",
            vt_symbol=args.vt_symbol,
            connect_setting=sanitize_setting(connect_setting),
        )
        main_engine.connect(connect_setting, DEFAULT_GATEWAY_NAME)

        contract = wait_for_contract(
            main_engine=main_engine,
            observer=observer,
            vt_symbol=args.vt_symbol,
            timeout=args.timeout,
            logger=logger,
        )

        symbol, exchange = extract_vt_symbol(args.vt_symbol)
        if contract.symbol != symbol or contract.exchange != exchange:
            log_event(
                logger,
                logging.WARNING,
                "okx.contract_symbol_mismatch",
                "Contract metadata differs from vt_symbol split result; continue with vt_symbol split result",
                contract_symbol=contract.symbol,
                split_symbol=symbol,
                contract_exchange=contract.exchange,
                split_exchange=exchange,
            )

        unique_bars, chunk_results, duplicate_count = query_history_by_chunks(
            main_engine=main_engine,
            gateway_name=DEFAULT_GATEWAY_NAME,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            chunks=chunks,
            logger=logger,
        )

        if not unique_bars:
            raise HistoryDownloadError(
                f"没有下载到任何历史数据: vt_symbol={args.vt_symbol}, interval={interval.value}, "
                f"start={start_dt.isoformat()}, end={end_dt.isoformat()}"
            )

        gaps = detect_time_gaps(unique_bars, interval_delta)
        if gaps:
            for gap in gaps[:20]:
                log_event(
                    logger,
                    logging.WARNING,
                    "history.gap_detected",
                    "Detected obvious history gap",
                    previous_dt=gap.previous_dt,
                    current_dt=gap.current_dt,
                    missing_bars=gap.missing_bars,
                    delta_seconds=gap.delta_seconds,
                )
            if len(gaps) > 20:
                log_event(
                    logger,
                    logging.WARNING,
                    "history.gap_summary",
                    "More gaps were detected than printed",
                    total_gaps=len(gaps),
                    printed=20,
                )
        else:
            log_event(
                logger,
                logging.INFO,
                "history.no_gap",
                "No obvious time gaps detected",
                vt_symbol=args.vt_symbol,
                interval=interval.value,
            )

        save_bars_to_database(unique_bars, logger)

        csv_path: Path | None = None
        if args.csv_copy:
            csv_path = export_csv_copy(
                vt_symbol=args.vt_symbol,
                interval=args.interval,
                start=start_dt,
                end=end_dt,
                bars=unique_bars,
            )
            log_event(
                logger,
                logging.INFO,
                "history.csv_exported",
                "Exported CSV copy",
                csv_path=csv_path,
                count=len(unique_bars),
            )

        instrument_payload = extract_contract_payload(contract)
        instrument_path = build_instrument_output_path(args.vt_symbol)
        save_contract_payload(instrument_payload, instrument_path)
        log_event(
            logger,
            logging.INFO,
            "instrument.updated",
            "Updated local instrument metadata file",
            instrument_path=instrument_path,
            contract=instrument_payload,
        )

        summary = build_final_summary(
            vt_symbol=args.vt_symbol,
            interval=args.interval,
            requested_start=start_dt,
            requested_end=end_dt,
            bars=unique_bars,
            raw_total_count=sum(result.count for result in chunk_results),
            duplicate_count=duplicate_count,
            gaps=gaps,
            chunk_results=chunk_results,
            csv_path=csv_path,
            instrument_path=instrument_path,
        )

        print_json_block(
            "History download summary:",
            {
                "vt_symbol": summary.vt_symbol,
                "interval": summary.interval,
                "requested_start": summary.requested_start,
                "requested_end": summary.requested_end,
                "total_raw_count": summary.total_raw_count,
                "total_unique_count": summary.total_unique_count,
                "duplicate_count": summary.duplicate_count,
                "first_dt": summary.first_dt,
                "last_dt": summary.last_dt,
                "gap_count": summary.gap_count,
                "csv_path": summary.csv_path,
                "instrument_path": summary.instrument_path,
                "chunk_results": [
                    {
                        "index": result.index,
                        "start": result.start,
                        "end": result.end,
                        "count": result.count,
                        "first_dt": result.first_dt,
                        "last_dt": result.last_dt,
                    }
                    for result in summary.chunk_results
                ],
            },
        )
        return 0
    except ConfigurationError as exc:
        log_event(
            logger,
            logging.ERROR,
            "history.config_error",
            str(exc),
            env_file=ENV_FILE,
        )
        return 1
    except ConnectionError as exc:
        recent_logs = observer.recent_log_messages() if observer is not None else []
        log_event(
            logger,
            logging.ERROR,
            "history.connection_error",
            str(exc),
            recent_logs=recent_logs,
        )
        return 2
    except HistoryDownloadError as exc:
        recent_logs = observer.recent_log_messages() if observer is not None else []
        log_event(
            logger,
            logging.ERROR,
            "history.download_error",
            str(exc),
            recent_logs=recent_logs,
        )
        return 3
    except DatabaseSaveError as exc:
        log_event(
            logger,
            logging.ERROR,
            "history.database_error",
            str(exc),
        )
        return 4
    except KeyboardInterrupt:
        log_event(
            logger,
            logging.WARNING,
            "history.interrupted",
            "Interrupted by user",
        )
        return 130
    except Exception:
        logger.exception(
            "Unexpected error during OKX history download",
            extra={"event": "history.unexpected_error"},
        )
        return 5
    finally:
        if main_engine is not None:
            try:
                main_engine.close()
            except Exception:
                logger.exception(
                    "Failed to close MainEngine cleanly",
                    extra={"event": "history.close_error"},
                )


if __name__ == "__main__":
    raise SystemExit(main())
