#!/usr/bin/env python3
"""Download OKX history with chunk-level persistence, retry, and resume."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import socket
import ssl
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING, Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
from history_time_utils import (
    DEFAULT_TIMEZONE,
    ChunkPlan,
    HistoryRange,
    build_half_open_chunks,
    expected_bar_count,
    normalize_bar_datetime,
    parse_history_range,
)
from history_utils import (
    HistoryCoverageSummary,
    MissingRange,
    analyze_history_coverage,
    build_instrument_config_path,
    build_repair_command,
    parse_interval,
    to_database_query_range,
    verify_database_coverage,
)

if TYPE_CHECKING:
    from vnpy.event import Event
    from vnpy.trader.constant import Exchange, Interval
    from vnpy.trader.database import BaseDatabase
    from vnpy.trader.object import BarData, ContractData, LogData


ENV_FILE: Path = PROJECT_ROOT / ".env"
DEFAULT_GATEWAY_NAME: str = "OKX"
DEFAULT_VT_SYMBOL: str = "BTCUSDT_SWAP_OKX.GLOBAL"
DEFAULT_INTERVAL: str = "1m"
DEFAULT_DATABASE_NAME: str = "sqlite"
DEFAULT_DATABASE_FILE: str = "database.db"
DEFAULT_CHECKPOINT_DIR: Path = PROJECT_ROOT / "data" / "history_manifests"
OKX_HISTORY_URL: str = "https://www.okx.com/api/v5/market/history-candles"


class ConfigurationError(Exception):
    """Raised when local runtime or CLI configuration is invalid."""


class GatewayConnectionError(Exception):
    """Raised when OKX gateway bootstrap fails."""


class HistoryDownloadError(Exception):
    """Raised when one or more history download steps fail."""


class DatabaseSaveError(Exception):
    """Raised when bar data cannot be saved into vn.py database."""


class HistoryEmptyResultError(Exception):
    """Raised when a chunk request returns an unexpected empty payload."""


class OkxRestError(Exception):
    """Raised when OKX public REST returns an error response."""

    def __init__(
        self,
        status_code: int,
        code: str | None,
        message: str,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int = status_code
        self.code: str | None = code
        self.response_body: str | None = response_body


class SourceQueryError(Exception):
    """Raised when a source exhausts its retries for one chunk."""

    def __init__(
        self,
        source: str,
        attempts: int,
        error_class: str,
        error_message: str,
    ) -> None:
        super().__init__(error_message)
        self.source: str = source
        self.attempts: int = attempts
        self.error_class: str = error_class
        self.error_message: str = error_message


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
class ChunkResult:
    """One chunk execution summary."""

    index: int
    start: datetime
    end_exclusive: datetime
    end_display: datetime
    timezone: str
    status: str
    attempts: int
    source_used: str | None
    count: int
    first_dt: datetime | None
    last_dt: datetime | None
    error_class: str | None
    error_message: str | None


@dataclass(slots=True)
class GatewayContext:
    """Runtime objects required for gateway history queries."""

    main_engine: Any
    observer: "GatewayObserver"
    contract: "ContractData"


class ManifestManager:
    """Persist chunk status transitions with atomic JSON writes."""

    def __init__(
        self,
        path: Path,
        vt_symbol: str,
        interval_value: str,
        history_range: HistoryRange,
        chunk_days: int,
        source: str,
        enabled: bool = True,
    ) -> None:
        self.path: Path = path
        self.enabled: bool = enabled
        self.payload: dict[str, Any] = self._load_or_create(
            vt_symbol=vt_symbol,
            interval_value=interval_value,
            history_range=history_range,
            chunk_days=chunk_days,
            source=source,
        )

    def _load_or_create(
        self,
        vt_symbol: str,
        interval_value: str,
        history_range: HistoryRange,
        chunk_days: int,
        source: str,
    ) -> dict[str, Any]:
        now = utc_now().isoformat()
        payload = {
            "version": 1,
            "vt_symbol": vt_symbol,
            "interval": interval_value,
            "start": history_range.start.isoformat(),
            "end_exclusive": history_range.end_exclusive.isoformat(),
            "end_display": history_range.end_display.isoformat(),
            "start_utc": history_range.start_utc.isoformat(),
            "end_exclusive_utc": history_range.end_exclusive_utc.isoformat(),
            "timezone": history_range.timezone_name,
            "chunk_days": chunk_days,
            "source": source,
            "created_at": now,
            "updated_at": now,
            "chunks": [],
        }

        if not self.path.exists():
            return payload

        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(f"checkpoint 文件损坏，无法读取: {self.path} | {exc!r}") from exc

        if not isinstance(loaded, dict):
            raise ConfigurationError(f"checkpoint 文件格式错误: {self.path}")

        payload.update(loaded)
        payload["vt_symbol"] = vt_symbol
        payload["interval"] = interval_value
        payload["start"] = history_range.start.isoformat()
        payload["end_exclusive"] = history_range.end_exclusive.isoformat()
        payload["end_display"] = history_range.end_display.isoformat()
        payload["start_utc"] = history_range.start_utc.isoformat()
        payload["end_exclusive_utc"] = history_range.end_exclusive_utc.isoformat()
        payload["timezone"] = history_range.timezone_name
        payload["chunk_days"] = chunk_days
        payload["source"] = source
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        payload.setdefault("chunks", [])

        stale_found = False
        for chunk in payload["chunks"]:
            if chunk.get("status") == "downloading":
                chunk["status"] = "pending"
                chunk["last_error"] = "stale_downloading_state"
                stale_found = True

        if stale_found:
            self._write_payload(payload)
        return payload

    def sync_chunks(self, chunks: list[ChunkPlan]) -> None:
        """Sync the manifest chunk list to the current plan."""

        existing: dict[tuple[str, str], dict[str, Any]] = {}
        for item in self.payload.get("chunks", []):
            if isinstance(item, dict):
                start_value = str(item.get("start", ""))
                end_value = str(item.get("end_exclusive") or item.get("end") or "")
                existing[(start_value, end_value)] = dict(item)

        synced: list[dict[str, Any]] = []
        for chunk in chunks:
            key = (chunk.start.isoformat(), chunk.end_exclusive.isoformat())
            legacy_key = (chunk.start.isoformat(), chunk.end_display.isoformat())
            current = existing.get(key) or existing.get(legacy_key, {})
            synced.append(
                {
                    "index": chunk.index,
                    "start": chunk.start.isoformat(),
                    "end_exclusive": chunk.end_exclusive.isoformat(),
                    "end_display": chunk.end_display.isoformat(),
                    "start_utc": chunk.start_utc.isoformat(),
                    "end_exclusive_utc": chunk.end_exclusive_utc.isoformat(),
                    "timezone": chunk.timezone_name,
                    "status": current.get("status", "pending"),
                    "attempts": int(current.get("attempts", 0) or 0),
                    "source_used": current.get("source_used"),
                    "bar_count": int(current.get("bar_count", 0) or 0),
                    "first_dt": current.get("first_dt"),
                    "last_dt": current.get("last_dt"),
                    "last_error": current.get("last_error"),
                    "saved_at": current.get("saved_at"),
                    "verified_at": current.get("verified_at"),
                }
            )

        self.payload["chunks"] = synced
        self.write()

    def get_chunk(self, plan: ChunkPlan) -> dict[str, Any]:
        """Return the manifest entry for a chunk."""

        for item in self.payload["chunks"]:
            if (
                item["start"] == plan.start.isoformat()
                and item["end_exclusive"] == plan.end_exclusive.isoformat()
            ):
                return item
        raise KeyError(f"Chunk not found in manifest: {plan}")

    def write(self) -> None:
        """Write the current payload to disk atomically."""

        self._write_payload(self.payload)

    def _write_payload(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = utc_now().isoformat()
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def update_chunk(self, plan: ChunkPlan, status: str | None = None, **fields: Any) -> dict[str, Any]:
        """Update one chunk record and flush it immediately."""

        item = self.get_chunk(plan)
        item["index"] = plan.index
        if status is not None:
            item["status"] = status

        for key, value in fields.items():
            item[key] = value.isoformat() if isinstance(value, datetime) else value

        self.write()
        return item


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

        message_lower = item["msg"].lower()
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
        description="Download OKX history data into vn.py sqlite database with resume support."
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
        help="Inclusive start datetime in ISO format. Date-only means 00:00:00 in --timezone.",
    )
    parser.add_argument(
        "--end",
        default="2026-03-31",
        help="Inclusive end datetime in ISO format. Date-only means the whole natural day in the configured timezone.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"History timezone. Default: {DEFAULT_TIMEZONE}.",
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
        help="Export a CSV copy into data/raw/ after database verification.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Gateway bootstrap and REST request timeout in seconds. Default: 30.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from a checkpoint manifest. Default: enabled.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR.relative_to(PROJECT_ROOT)),
        help="Checkpoint directory. Default: data/history_manifests/.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "gateway", "rest"),
        default="auto",
        help="History source selection. Default: auto.",
    )
    parser.add_argument(
        "--save-per-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each successful chunk immediately. Default: enabled.",
    )
    parser.add_argument(
        "--verify-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify each saved chunk in sqlite immediately. Default: enabled.",
    )
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help="Only download missing database ranges and attempt automatic gap repair.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum retries per source and per chunk. Default: 8.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=2.0,
        help="Base backoff delay in seconds. Default: 2.0.",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=120.0,
        help="Maximum retry delay in seconds. Default: 120.0.",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.35,
        help="Sleep after each source request to reduce rate limit pressure. Default: 0.35.",
    )
    parser.add_argument(
        "--strict-completeness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when final sqlite coverage still has gaps. Default: enabled.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit with code 0 even when some chunks still fail or gaps remain.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan and missing ranges without contacting OKX or writing data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def utc_now() -> datetime:
    """Return current UTC time."""

    return datetime.now(timezone.utc)


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

    required_fields = ("OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE", "OKX_SERVER")
    missing_fields = [field for field in required_fields if not values.get(field)]
    if server_override:
        missing_fields = [field for field in missing_fields if field != "OKX_SERVER"]

    if missing_fields:
        raise ConfigurationError(f".env 缺字段或为空: {', '.join(missing_fields)}")

    proxy_port_raw = values.get("OKX_PROXY_PORT", "0") or "0"
    try:
        proxy_port = int(proxy_port_raw)
    except ValueError as exc:
        raise ConfigurationError(f"OKX_PROXY_PORT 必须是整数，当前值为: {proxy_port_raw}") from exc

    proxy_host = values.get("OKX_PROXY_HOST", "")
    if proxy_host and proxy_port <= 0:
        raise ConfigurationError("代理配置错误：OKX_PROXY_HOST 已设置，但 OKX_PROXY_PORT <= 0")
    if proxy_port > 0 and not proxy_host:
        raise ConfigurationError("代理配置错误：OKX_PROXY_PORT 已设置，但 OKX_PROXY_HOST 为空")

    server_value = (server_override or values.get("OKX_SERVER", "")).upper()
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

    setting = {key: coerce_default_value(value) for key, value in default_setting.items()}
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


def infer_connection_failure(observer: GatewayObserver, vt_symbol: str) -> str:
    """Infer clearer connection failure reason from recent vn.py gateway logs."""

    combined_logs = "\n".join(item["msg"] for item in observer.recent_logs)
    logs_lower = combined_logs.lower()

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

    deadline = time.monotonic() + timeout
    next_progress_log = time.monotonic() + 5.0
    contract_detected_at: float | None = None
    contract_grace_seconds = 3.0

    while time.monotonic() < deadline:
        contract = observer.contract or main_engine.get_contract(vt_symbol)
        if contract is not None:
            observer.contract = contract
            observer.contract_ready.set()
            if contract_detected_at is None:
                contract_detected_at = time.monotonic()

        if observer.private_login_failed:
            raise GatewayConnectionError(
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

    raise GatewayConnectionError(infer_connection_failure(observer, vt_symbol))


def initialize_gateway_context(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> GatewayContext:
    """Connect vnpy_okx and wait until the target contract is available."""

    from vnpy.event import EventEngine
    from vnpy.trader.engine import MainEngine
    from vnpy.trader.event import EVENT_CONTRACT, EVENT_LOG
    from vnpy_okx import OkxGateway

    env_config = read_env_config(args.server)
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
    return GatewayContext(main_engine=main_engine, observer=observer, contract=contract)


def build_chunks_for_missing_ranges(
    ranges: list[MissingRange],
    interval_delta: timedelta,
    chunk_days: int,
    timezone_name: str,
) -> list[ChunkPlan]:
    """Split missing ranges into fixed-size half-open chunks."""

    chunks: list[ChunkPlan] = []
    index = 1
    for missing_range in ranges:
        missing_history_range = HistoryRange.from_bounds(
            start=missing_range.start,
            end_exclusive=missing_range.end + interval_delta,
            interval_delta=interval_delta,
            timezone_name=timezone_name,
        )
        for chunk in build_half_open_chunks(missing_history_range, chunk_days=chunk_days):
            chunks.append(
                ChunkPlan(
                    index=index,
                    start=chunk.start,
                    end_exclusive=chunk.end_exclusive,
                    end_display=chunk.end_display,
                    timezone_name=chunk.timezone_name,
                )
            )
            index += 1
    return chunks


def safe_filename_token(value: str) -> str:
    """Convert a CLI token into a filesystem-safe filename component."""

    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def build_manifest_path(
    checkpoint_dir: Path,
    vt_symbol: str,
    interval_value: str,
    start_arg: str,
    end_arg: str,
    timezone_name: str,
) -> Path:
    """Build the JSON checkpoint path for a download task."""

    filename = (
        f"{safe_filename_token(vt_symbol)}_"
        f"{safe_filename_token(interval_value)}_"
        f"{safe_filename_token(start_arg)}_"
        f"{safe_filename_token(end_arg)}_"
        f"{safe_filename_token(timezone_name)}.json"
    )
    return checkpoint_dir / filename


def classify_history_error(exc_or_response: object) -> str:
    """Classify source and storage errors into retry/fatal buckets."""

    status_code: int | None = None
    code: str | None = None

    if isinstance(exc_or_response, OkxRestError):
        status_code = exc_or_response.status_code
        code = exc_or_response.code
        message = str(exc_or_response).lower()
    elif isinstance(exc_or_response, dict):
        status_code = int(exc_or_response.get("status_code", 0) or 0) or None
        raw_code = exc_or_response.get("code")
        code = str(raw_code) if raw_code is not None else None
        message = str(exc_or_response.get("message", "")).lower()
    else:
        message = str(exc_or_response).lower()

    if isinstance(exc_or_response, HistoryEmptyResultError):
        return "retryable_empty_result"
    if isinstance(exc_or_response, (TimeoutError, socket.timeout)):
        return "retryable_network"
    if isinstance(exc_or_response, (ConnectionError, ConnectionResetError, ssl.SSLError, URLError)):
        return "retryable_network"
    if status_code == 429 or code == "50011" or "50011" in message or "rate limit" in message:
        return "retryable_rate_limit"
    if code in {"50013", "50001"} or "50013" in message or "50001" in message:
        return "retryable_exchange_busy"
    if any(
        token in message
        for token in ("temporarily unavailable", "systems are busy", "service temporarily unavailable")
    ):
        return "retryable_exchange_busy"
    if any(
        token in message
        for token in ("api key", "secret", "passphrase", "login failed", "authentication", "demo / real")
    ):
        return "fatal_auth"
    if any(
        token in message
        for token in (".env", "proxy", "vt_symbol 格式无效", "unsupported interval", "无法解析", "配置错误")
    ):
        return "fatal_config"
    if any(
        token in message
        for token in ("symbol not found", "invalid symbol", "contract not found", "无法推导 okx instid")
    ):
        return "fatal_symbol"
    if "database" in message or "sqlite" in message:
        return "fatal_database"
    return "unknown"


def should_retry_history_error(error_class: str) -> bool:
    """Return whether a classified source error should be retried."""

    return error_class in {
        "retryable_network",
        "retryable_rate_limit",
        "retryable_exchange_busy",
        "retryable_empty_result",
        "unknown",
    }


def compute_backoff_sleep(
    attempt: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> float:
    """Compute exponential backoff with jitter."""

    base_delay = min(retry_max_delay, retry_base_delay * (2 ** (attempt - 1)))
    return base_delay * random.uniform(0.8, 1.3)


def clone_bar(bar: BarData) -> BarData:
    """Clone a vn.py BarData object before sqlite mutates its fields."""

    from vnpy.trader.object import BarData

    return BarData(
        gateway_name=bar.gateway_name,
        symbol=bar.symbol,
        exchange=bar.exchange,
        datetime=bar.datetime,
        interval=bar.interval,
        volume=bar.volume,
        turnover=bar.turnover,
        open_interest=bar.open_interest,
        open_price=bar.open_price,
        high_price=bar.high_price,
        low_price=bar.low_price,
        close_price=bar.close_price,
    )


def normalize_chunk_bars(
    bars: list[BarData],
    chunk: ChunkPlan,
    timezone_name: str,
    interval_delta: timedelta,
) -> tuple[list[BarData], int]:
    """Normalize, deduplicate, and clamp bars to one half-open chunk."""

    bars_by_dt: dict[datetime, BarData] = {}
    duplicate_count = 0
    for bar in bars:
        bar_dt = normalize_bar_datetime(
            bar.datetime,
            timezone_name=timezone_name,
            interval_delta=interval_delta,
        )
        if bar_dt < chunk.start or bar_dt >= chunk.end_exclusive:
            continue

        normalized_bar = clone_bar(bar)
        normalized_bar.datetime = bar_dt

        if bar_dt in bars_by_dt:
            duplicate_count += 1
        bars_by_dt[bar_dt] = normalized_bar

    return [bars_by_dt[item] for item in sorted(bars_by_dt)], duplicate_count


def combine_source_used(current_source: str | None, new_source: str | None) -> str | None:
    """Combine multiple source usages into one manifest string."""

    if not current_source:
        return new_source
    if not new_source or current_source == new_source:
        return current_source
    return "mixed"


def extract_recent_gateway_message(observer: GatewayObserver) -> str:
    """Return recent gateway logs as a compact diagnostic string."""

    messages = [item["msg"] for item in observer.recent_log_messages(limit=12) if item.get("msg")]
    return " | ".join(messages[-6:])


def query_gateway_chunk(
    gateway_context: GatewayContext,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    chunk: ChunkPlan,
    timezone_name: str,
    interval_delta: timedelta,
) -> list[BarData]:
    """Query one chunk through vn.py MainEngine.query_history."""

    from vnpy.trader.object import HistoryRequest

    request = HistoryRequest(
        symbol=symbol,
        exchange=exchange,
        start=chunk.start,
        end=chunk.end_exclusive,
        interval=interval,
    )
    bars = gateway_context.main_engine.query_history(request, DEFAULT_GATEWAY_NAME)
    normalized_bars, _duplicate_count = normalize_chunk_bars(
        bars,
        chunk=chunk,
        timezone_name=timezone_name,
        interval_delta=interval_delta,
    )
    if normalized_bars:
        return normalized_bars

    recent_logs = extract_recent_gateway_message(gateway_context.observer)
    raise HistoryEmptyResultError(
        f"Gateway returned empty history result for chunk {chunk.index}: "
        f"{chunk.start.isoformat()} -> {chunk.end_exclusive.isoformat()} | recent_logs={recent_logs}"
    )


def okx_interval_value(interval: Interval) -> str:
    """Map vn.py Interval into OKX public REST bar parameter."""

    mapping = {
        "1m": "1m",
        "1h": "1H",
        "d": "1D",
        "w": "1W",
    }
    try:
        return mapping[interval.value]
    except KeyError as exc:
        raise ConfigurationError(f"REST fallback 暂不支持 interval={interval.value}") from exc


def derive_okx_inst_id(symbol: str) -> str:
    """Derive OKX instId from vn.py symbol when no local config file is available."""

    raw_symbol = symbol.upper()
    if raw_symbol.endswith("_OKX"):
        raw_symbol = raw_symbol[:-4]

    suffixes = ("_SWAP", "_SPOT", "_FUTURES")
    product_suffix = next((item for item in suffixes if raw_symbol.endswith(item)), None)
    if product_suffix is None:
        raise ConfigurationError(f"无法推导 OKX instId，symbol 缺少产品后缀: {symbol}")

    product = product_suffix.lstrip("_")
    base_quote = raw_symbol[: -len(product_suffix)]

    for quote in ("USDT", "USDC", "USD", "BTC", "ETH", "EUR"):
        if base_quote.endswith(quote) and len(base_quote) > len(quote):
            base = base_quote[: -len(quote)]
            if product == "SPOT":
                return f"{base}-{quote}"
            return f"{base}-{quote}-{product}"

    raise ConfigurationError(f"无法推导 OKX instId，未知报价币后缀: {symbol}")


def resolve_okx_inst_id(vt_symbol: str, contract: ContractData | None) -> str:
    """Resolve OKX instId from live contract, local config, or symbol heuristics."""

    if contract is not None and getattr(contract, "name", ""):
        return str(contract.name)

    instrument_path = build_instrument_config_path(vt_symbol)
    if instrument_path.exists():
        try:
            payload = json.loads(instrument_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(f"读取 instrument 配置失败: {instrument_path} | {exc!r}") from exc

        for key in ("name", "instId", "symbol"):
            value = payload.get(key)
            if isinstance(value, str) and "-" in value:
                return value

    from vnpy.trader.utility import extract_vt_symbol

    symbol, _exchange = extract_vt_symbol(vt_symbol)
    return derive_okx_inst_id(symbol)


def parse_okx_rest_bar(
    row: list[Any],
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    timezone_name: str,
    interval_delta: timedelta,
) -> BarData | None:
    """Convert OKX public REST candle row into vn.py BarData."""

    from vnpy.trader.object import BarData

    if len(row) < 6:
        return None

    confirm = str(row[-1])
    if confirm != "1":
        return None

    ts = int(str(row[0]))
    turnover_text = row[7] if len(row) > 7 else row[6] if len(row) > 6 else 0
    return BarData(
        gateway_name="OKX_REST",
        symbol=symbol,
        exchange=exchange,
        datetime=normalize_bar_datetime(
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            timezone_name=timezone_name,
            interval_delta=interval_delta,
        ),
        interval=interval,
        volume=float(row[5] or 0),
        turnover=float(turnover_text or 0),
        open_interest=0.0,
        open_price=float(row[1] or 0),
        high_price=float(row[2] or 0),
        low_price=float(row[3] or 0),
        close_price=float(row[4] or 0),
    )


def request_okx_history_page(
    inst_id: str,
    interval: Interval,
    after: str,
    timeout: float,
    throttle_seconds: float,
) -> list[list[Any]]:
    """Request one page from OKX public history-candles REST API."""

    params = {
        "instId": inst_id,
        "bar": okx_interval_value(interval),
        "after": after,
        "limit": "100",
    }
    request = Request(
        url=f"{OKX_HISTORY_URL}?{urlencode(params)}",
        headers={"User-Agent": "cta-history-downloader/1.0"},
    )

    response_text = ""
    try:
        with urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
            status_code = int(getattr(response, "status", response.getcode()))
    except HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        code: str | None = None
        message = exc.reason if isinstance(exc.reason, str) else str(exc)
        try:
            payload = json.loads(response_text)
            if isinstance(payload, dict):
                raw_code = payload.get("code")
                code = str(raw_code) if raw_code is not None else None
                message = str(payload.get("msg") or message)
        except Exception:
            pass
        raise OkxRestError(exc.code, code, message, response_body=response_text) from exc
    finally:
        time.sleep(max(throttle_seconds, 0.0))

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OkxRestError(200, None, "OKX REST 返回了无法解析的 JSON", response_body=response_text) from exc

    if not isinstance(payload, dict):
        raise OkxRestError(200, None, "OKX REST 返回格式错误", response_body=response_text)

    raw_code = payload.get("code")
    code = str(raw_code) if raw_code is not None else None
    if code not in (None, "", "0"):
        raise OkxRestError(
            status_code=200,
            code=code,
            message=str(payload.get("msg") or "OKX REST returned an error code"),
            response_body=response_text,
        )

    data = payload.get("data")
    if not isinstance(data, list):
        raise OkxRestError(200, code, "OKX REST data 字段缺失或格式错误", response_body=response_text)
    return data


def query_rest_chunk(
    vt_symbol: str,
    contract: ContractData | None,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    chunk: ChunkPlan,
    timezone_name: str,
    interval_delta: timedelta,
    timeout: float,
    throttle_seconds: float,
) -> list[BarData]:
    """Query one chunk through OKX public REST history-candles."""

    inst_id = resolve_okx_inst_id(vt_symbol, contract)
    after = str(int(chunk.end_exclusive.astimezone(timezone.utc).timestamp() * 1000))
    seen_after_values: set[str] = set()
    bars_by_dt: dict[datetime, BarData] = {}

    while True:
        page = request_okx_history_page(
            inst_id=inst_id,
            interval=interval,
            after=after,
            timeout=timeout,
            throttle_seconds=throttle_seconds,
        )
        if not page:
            break

        for row in page:
            bar = parse_okx_rest_bar(
                row,
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                timezone_name=timezone_name,
                interval_delta=interval_delta,
            )
            if bar is None:
                continue
            if chunk.start <= bar.datetime < chunk.end_exclusive:
                bars_by_dt[bar.datetime] = bar

        oldest_ts = str(page[-1][0])
        oldest_dt = normalize_bar_datetime(
            datetime.fromtimestamp(int(oldest_ts) / 1000, tz=timezone.utc),
            timezone_name=timezone_name,
            interval_delta=interval_delta,
        )
        if oldest_dt <= chunk.start:
            break
        if oldest_ts in seen_after_values:
            break
        seen_after_values.add(oldest_ts)
        after = oldest_ts

    bars = [bars_by_dt[item] for item in sorted(bars_by_dt)]
    if bars:
        return bars

    raise HistoryEmptyResultError(
        f"REST returned empty history result for chunk {chunk.index}: "
        f"{chunk.start.isoformat()} -> {chunk.end_exclusive.isoformat()} | instId={inst_id}"
    )


def query_source_with_retry(
    source: str,
    chunk: ChunkPlan,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    logger: logging.Logger,
    fetcher: Callable[[ChunkPlan], list[BarData]],
) -> tuple[list[BarData], int]:
    """Run one source with retry/backoff/jitter for a specific chunk."""

    if max_retries <= 0:
        raise ConfigurationError(f"max-retries 必须大于 0，当前值为: {max_retries}")

    for attempt in range(1, max_retries + 1):
        try:
            bars = fetcher(chunk)
            if not bars:
                raise HistoryEmptyResultError(
                    f"Source={source} returned no bars for chunk {chunk.index}: "
                    f"{chunk.start.isoformat()} -> {chunk.end_exclusive.isoformat()}"
                )
            return bars, attempt
        except Exception as exc:
            error_class = classify_history_error(exc)
            error_message = str(exc)
            if attempt >= max_retries or not should_retry_history_error(error_class):
                raise SourceQueryError(
                    source=source,
                    attempts=attempt,
                    error_class=error_class,
                    error_message=error_message,
                ) from exc

            sleep_seconds = compute_backoff_sleep(
                attempt=attempt,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
            )
            log_event(
                logger,
                logging.WARNING,
                "history.chunk_retry",
                "Retrying failed history chunk request",
                chunk_index=chunk.index,
                attempt=attempt,
                max_retries=max_retries,
                source=source,
                error_class=error_class,
                error_message=error_message,
                sleep_seconds=sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise AssertionError("unreachable")


def query_chunk_with_retry(
    args: argparse.Namespace,
    chunk: ChunkPlan,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    interval_delta: timedelta,
    gateway_context: GatewayContext | None,
    logger: logging.Logger,
) -> tuple[list[BarData], str, int]:
    """Query one chunk using gateway, REST, or auto fallback."""

    total_attempts = 0
    last_error: SourceQueryError | None = None

    if args.source in ("gateway", "auto"):
        if gateway_context is None:
            if args.source == "gateway":
                raise HistoryDownloadError("source=gateway 但 OKX gateway 未成功初始化。")
        else:
            try:
                bars, attempts = query_source_with_retry(
                    source="gateway",
                    chunk=chunk,
                    max_retries=args.max_retries,
                    retry_base_delay=args.retry_base_delay,
                    retry_max_delay=args.retry_max_delay,
                    logger=logger,
                    fetcher=lambda current_chunk: query_gateway_chunk(
                        gateway_context=gateway_context,
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        chunk=current_chunk,
                        timezone_name=args.timezone,
                        interval_delta=interval_delta,
                    ),
                )
                return bars, "gateway", attempts
            except SourceQueryError as exc:
                total_attempts += exc.attempts
                last_error = exc
                if args.source == "gateway" or exc.error_class == "fatal_symbol":
                    raise HistoryDownloadError(
                        f"gateway chunk 查询失败: chunk={chunk.index}, error_class={exc.error_class}, "
                        f"message={exc.error_message}"
                    ) from exc

                log_event(
                    logger,
                    logging.WARNING,
                    "history.fallback_rest",
                    "Gateway failed after retries, falling back to OKX public REST",
                    chunk_index=chunk.index,
                    error_class=exc.error_class,
                    error_message=exc.error_message,
                )

    if args.source in ("rest", "auto"):
        try:
            bars, attempts = query_source_with_retry(
                source="rest",
                chunk=chunk,
                max_retries=args.max_retries,
                retry_base_delay=args.retry_base_delay,
                retry_max_delay=args.retry_max_delay,
                logger=logger,
                fetcher=lambda current_chunk: query_rest_chunk(
                    vt_symbol=args.vt_symbol,
                    contract=gateway_context.contract if gateway_context is not None else None,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    chunk=current_chunk,
                    timezone_name=args.timezone,
                    interval_delta=interval_delta,
                    timeout=args.timeout,
                    throttle_seconds=args.throttle_seconds,
                ),
            )
            return bars, "rest", total_attempts + attempts
        except SourceQueryError as exc:
            total_attempts += exc.attempts
            raise HistoryDownloadError(
                f"REST chunk 查询失败: chunk={chunk.index}, error_class={exc.error_class}, "
                f"message={exc.error_message}"
            ) from exc

    if last_error is not None:
        raise HistoryDownloadError(
            f"历史数据下载失败: chunk={chunk.index}, source={last_error.source}, "
            f"error_class={last_error.error_class}, message={last_error.error_message}"
        ) from last_error

    raise HistoryDownloadError(f"没有可用的数据源处理 chunk={chunk.index}")


def is_retryable_database_lock(exc: Exception) -> bool:
    """Return True when sqlite is temporarily locked."""

    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def get_database_instance() -> BaseDatabase:
    """Create the vn.py sqlite database instance."""

    from vnpy.trader.database import get_database
    from vnpy.trader.setting import SETTINGS

    try:
        return get_database()
    except Exception as exc:
        raise DatabaseSaveError(
            "创建数据库实例失败。请检查数据库驱动或 SETTINGS 配置，不是策略问题。"
            f" 当前配置: database.name={SETTINGS.get('database.name')}, "
            f"database.database={SETTINGS.get('database.database')} | {exc!r}"
        ) from exc


def save_bars_to_database(
    database: BaseDatabase,
    bars: list[BarData],
    logger: logging.Logger,
    max_attempts: int = 3,
) -> None:
    """Save deduplicated bars into vn.py local database with lock retry."""

    if not bars:
        raise DatabaseSaveError("没有可写入数据库的 bar 数据")

    for attempt in range(1, max_attempts + 1):
        bars_copy = [clone_bar(bar) for bar in bars]
        try:
            result = database.save_bar_data(bars_copy, stream=False)
        except Exception as exc:
            if attempt < max_attempts and is_retryable_database_lock(exc):
                sleep_seconds = compute_backoff_sleep(attempt, retry_base_delay=0.5, retry_max_delay=5.0)
                log_event(
                    logger,
                    logging.WARNING,
                    "db.save_retry",
                    "Retrying sqlite save after temporary database lock",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    sleep_seconds=sleep_seconds,
                    error_message=str(exc),
                )
                time.sleep(sleep_seconds)
                continue
            raise DatabaseSaveError(
                "save_bar_data 执行异常。优先检查数据库驱动和 SETTINGS 配置，不是策略问题。"
                f" 当前数据库实例: {type(database).__name__} | {exc!r}"
            ) from exc

        if not result:
            raise DatabaseSaveError("save_bar_data 返回 False。请检查数据库驱动和 SETTINGS 配置。")

        log_event(
            logger,
            logging.INFO,
            "db.save_success",
            "Saved chunk bars into vn.py database",
            database_type=type(database).__name__,
            count=len(bars),
            first_dt=bars[0].datetime,
            last_dt=bars[-1].datetime,
        )
        return

    raise AssertionError("unreachable")


def load_database_bars(
    database: BaseDatabase,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    history_range: HistoryRange,
    logger: logging.Logger,
    max_attempts: int = 3,
) -> list[BarData]:
    """Load one range from sqlite and retry on temporary locks."""

    query_start, query_end = to_database_query_range(history_range)

    for attempt in range(1, max_attempts + 1):
        try:
            return database.load_bar_data(symbol, exchange, interval, query_start, query_end)
        except Exception as exc:
            if attempt < max_attempts and is_retryable_database_lock(exc):
                sleep_seconds = compute_backoff_sleep(attempt, retry_base_delay=0.5, retry_max_delay=5.0)
                log_event(
                    logger,
                    logging.WARNING,
                    "db.load_retry",
                    "Retrying sqlite load after temporary database lock",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    sleep_seconds=sleep_seconds,
                    error_message=str(exc),
                )
                time.sleep(sleep_seconds)
                continue
            raise DatabaseSaveError(f"load_bar_data 执行异常: {exc!r}") from exc

    raise AssertionError("unreachable")


def verify_chunk_in_database(
    database: BaseDatabase,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    chunk: ChunkPlan,
    interval_delta: timedelta,
    timezone_name: str,
) -> HistoryCoverageSummary:
    """Verify one chunk already saved into sqlite."""

    history_range = HistoryRange.from_bounds(
        start=chunk.start,
        end_exclusive=chunk.end_exclusive,
        interval_delta=interval_delta,
        timezone_name=timezone_name,
    )
    return verify_database_coverage(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        history_range=history_range,
        database=database,
    )


def export_csv_copy(
    vt_symbol: str,
    interval_value: str,
    start: datetime,
    end: datetime,
    bars: list[BarData],
) -> Path:
    """Export a CSV copy of bar data into data/raw."""

    symbol_part = vt_symbol.split(".", maxsplit=1)[0].lower()
    filename = (
        f"{symbol_part}_{interval_value}_"
        f"{start.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{end.strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    output_path = PROJECT_ROOT / "data" / "raw" / filename
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
                    "interval": bar.interval.value if bar.interval else "",
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

    payload: dict[str, Any] = {}
    for field_name in (
        "vt_symbol",
        "symbol",
        "exchange",
        "name",
        "size",
        "pricetick",
        "min_volume",
        "gateway_name",
        "history_data",
    ):
        if hasattr(contract, field_name):
            payload[field_name] = to_jsonable(getattr(contract, field_name))
    return payload


def save_contract_payload(payload: dict[str, Any], output_path: Path) -> None:
    """Write contract metadata JSON to config/instruments."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def log_coverage_summary(
    logger: logging.Logger,
    event_name: str,
    message: str,
    coverage: HistoryCoverageSummary,
) -> None:
    """Emit a structured coverage summary log."""

    largest_gap = (
        {
            "start": coverage.largest_gap.start,
            "end": coverage.largest_gap.end,
            "missing_count": coverage.largest_gap.missing_count,
        }
        if coverage.largest_gap is not None
        else None
    )
    log_event(
        logger,
        logging.INFO if coverage.is_complete else logging.WARNING,
        event_name,
        message,
        total_count=coverage.total_count,
        first_dt=coverage.first_dt,
        last_dt=coverage.last_dt,
        expected_count=coverage.expected_count,
        missing_count=coverage.missing_count,
        gap_count=coverage.gap_count,
        largest_gap=largest_gap,
        missing_ranges=[
            {"start": item.start, "end": item.end, "missing_count": item.missing_count}
            for item in coverage.missing_ranges[:20]
        ],
    )


def plan_download_ranges(
    args: argparse.Namespace,
    database: BaseDatabase,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    history_range: HistoryRange,
) -> tuple[HistoryCoverageSummary, list[MissingRange]]:
    """Plan download ranges based on current database coverage."""

    coverage = verify_database_coverage(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        history_range=history_range,
        database=database,
    )

    if args.repair_missing:
        ranges = coverage.missing_ranges
    else:
        ranges = [
            MissingRange(
                start=history_range.start,
                end=history_range.end_display,
                missing_count=coverage.expected_count or expected_bar_count(history_range),
            )
        ]

    return coverage, ranges


def load_final_bars(
    database: BaseDatabase,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    history_range: HistoryRange,
    logger: logging.Logger,
) -> list[BarData]:
    """Load the final requested range from sqlite for CSV export and summary."""

    return load_database_bars(
        database=database,
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        history_range=history_range,
        logger=logger,
    )


def chunk_result_from_manifest(item: dict[str, Any]) -> ChunkResult:
    """Convert a manifest chunk entry into a final summary object."""

    def parse_optional_dt(value: Any) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(str(value))

    return ChunkResult(
        index=int(item.get("index", 0) or 0),
        start=datetime.fromisoformat(str(item["start"])),
        end_exclusive=datetime.fromisoformat(str(item["end_exclusive"])),
        end_display=datetime.fromisoformat(str(item["end_display"])),
        timezone=str(item.get("timezone", DEFAULT_TIMEZONE)),
        status=str(item.get("status", "pending")),
        attempts=int(item.get("attempts", 0) or 0),
        source_used=item.get("source_used"),
        count=int(item.get("bar_count", 0) or 0),
        first_dt=parse_optional_dt(item.get("first_dt")),
        last_dt=parse_optional_dt(item.get("last_dt")),
        error_class=classify_history_error(item.get("last_error", "")) if item.get("last_error") else None,
        error_message=item.get("last_error"),
    )


def main() -> int:
    """Run OKX history download."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("download_okx_history", verbose=args.verbose)
    strict_mode = bool(args.strict_completeness and not args.allow_partial)

    gateway_context: GatewayContext | None = None

    try:
        from vnpy.trader.utility import extract_vt_symbol

        interval, interval_delta = parse_interval(args.interval)
        try:
            history_range = parse_history_range(
                start_arg=args.start,
                end_arg=args.end,
                interval_delta=interval_delta,
                timezone_name=args.timezone,
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc

        configure_sqlite_settings(logger)
        symbol, exchange = extract_vt_symbol(args.vt_symbol)
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = PROJECT_ROOT / checkpoint_dir
        manifest_path = build_manifest_path(
            checkpoint_dir=checkpoint_dir,
            vt_symbol=args.vt_symbol,
            interval_value=args.interval,
            start_arg=args.start,
            end_arg=args.end,
            timezone_name=args.timezone,
        )
        database = get_database_instance()

        existing_coverage, planned_ranges = plan_download_ranges(
            args=args,
            database=database,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            history_range=history_range,
        )
        if args.repair_missing:
            log_coverage_summary(
                logger,
                "history.initial_coverage",
                "Computed initial sqlite coverage before repair download",
                existing_coverage,
            )

        chunks = build_chunks_for_missing_ranges(
            ranges=planned_ranges,
            interval_delta=interval_delta,
            chunk_days=args.chunk_days,
            timezone_name=history_range.timezone_name,
        )
        manifest = ManifestManager(
            path=manifest_path,
            vt_symbol=args.vt_symbol,
            interval_value=args.interval,
            history_range=history_range,
            chunk_days=args.chunk_days,
            source=args.source,
            enabled=not args.dry_run,
        )
        manifest.sync_chunks(chunks)

        log_event(
            logger,
            logging.INFO,
            "history.download_start",
            "Starting OKX history download",
            vt_symbol=args.vt_symbol,
            interval=args.interval,
            timezone=history_range.timezone_name,
            start=history_range.start,
            end_exclusive=history_range.end_exclusive,
            end_display=history_range.end_display,
            start_utc=history_range.start_utc,
            end_exclusive_utc=history_range.end_exclusive_utc,
            chunk_days=args.chunk_days,
            chunk_count=len(chunks),
            timeout=args.timeout,
            source=args.source,
            resume=args.resume,
            repair_missing=args.repair_missing,
            save_per_chunk=args.save_per_chunk,
            verify_db=args.verify_db,
            strict_completeness=strict_mode,
            dry_run=args.dry_run,
            manifest_path=manifest_path,
        )

        if args.dry_run:
            print_json_block(
                "History download plan:",
                {
                    "vt_symbol": args.vt_symbol,
                    "interval": args.interval,
                    "timezone": history_range.timezone_name,
                    "start": history_range.start,
                    "end_exclusive": history_range.end_exclusive,
                    "end_display": history_range.end_display,
                    "start_utc": history_range.start_utc,
                    "end_exclusive_utc": history_range.end_exclusive_utc,
                    "expected_count": expected_bar_count(history_range),
                    "source": args.source,
                    "manifest_path": manifest_path,
                    "initial_coverage": {
                        "total_count": existing_coverage.total_count,
                        "first_dt": existing_coverage.first_dt,
                        "last_dt": existing_coverage.last_dt,
                        "expected_count": existing_coverage.expected_count,
                        "missing_count": existing_coverage.missing_count,
                        "gap_count": existing_coverage.gap_count,
                        "missing_ranges": [
                            {
                                "start": item.start,
                                "end": item.end,
                                "missing_count": item.missing_count,
                            }
                            for item in existing_coverage.missing_ranges
                        ],
                    },
                    "planned_chunks": [
                        {
                            "index": chunk.index,
                            "start": chunk.start,
                            "end_exclusive": chunk.end_exclusive,
                            "end_display": chunk.end_display,
                            "start_utc": chunk.start_utc,
                            "end_exclusive_utc": chunk.end_exclusive_utc,
                            "timezone": chunk.timezone_name,
                        }
                        for chunk in chunks
                    ],
                },
            )
            return 0

        if args.source in ("gateway", "auto"):
            try:
                gateway_context = initialize_gateway_context(args, logger)
            except Exception as exc:
                if args.source == "gateway":
                    raise
                log_event(
                    logger,
                    logging.WARNING,
                    "history.gateway_unavailable",
                    "Gateway bootstrap failed, continuing with REST fallback only",
                    error_class=classify_history_error(exc),
                    error_message=str(exc),
                )

        buffered_bars: list[BarData] = []
        chunk_results: list[ChunkResult] = []

        for chunk in chunks:
            entry = manifest.get_chunk(chunk)
            if args.resume and str(entry.get("status")) in {"verified", "skipped_existing"}:
                chunk_results.append(chunk_result_from_manifest(entry))
                continue

            if args.verify_db:
                current_coverage = verify_chunk_in_database(
                    database=database,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    chunk=chunk,
                    interval_delta=interval_delta,
                    timezone_name=history_range.timezone_name,
                )
                if current_coverage.is_complete:
                    entry = manifest.update_chunk(
                        chunk,
                        status="skipped_existing",
                        bar_count=current_coverage.total_count,
                        first_dt=current_coverage.first_dt,
                        last_dt=current_coverage.last_dt,
                        last_error=None,
                        verified_at=utc_now(),
                    )
                    chunk_results.append(chunk_result_from_manifest(entry))
                    continue
                if args.resume and str(entry.get("status")) == "verified":
                    log_event(
                        logger,
                        logging.WARNING,
                        "history.manifest_db_mismatch",
                        "Manifest marked chunk verified but sqlite coverage is incomplete; re-downloading chunk",
                        chunk_index=chunk.index,
                    )

            total_attempts = 0
            source_used: str | None = None
            first_dt: datetime | None = None
            last_dt: datetime | None = None
            saved_count = 0

            try:
                manifest.update_chunk(chunk, status="downloading", last_error=None)
                bars, source_used, attempts = query_chunk_with_retry(
                    args=args,
                    chunk=chunk,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    interval_delta=interval_delta,
                    gateway_context=gateway_context,
                    logger=logger,
                )
                total_attempts += attempts
                normalized_bars, _duplicate_count = normalize_chunk_bars(
                    bars,
                    chunk=chunk,
                    timezone_name=history_range.timezone_name,
                    interval_delta=interval_delta,
                )
                if not normalized_bars:
                    raise HistoryDownloadError(
                        f"chunk={chunk.index} 下载结果为空: "
                        f"{chunk.start.isoformat()} -> {chunk.end_exclusive.isoformat()}"
                    )

                first_dt = normalized_bars[0].datetime
                last_dt = normalized_bars[-1].datetime
                saved_count = len(normalized_bars)
                manifest.update_chunk(
                    chunk,
                    status="downloading",
                    attempts=total_attempts,
                    source_used=source_used,
                    bar_count=saved_count,
                    first_dt=first_dt,
                    last_dt=last_dt,
                    last_error=None,
                )

                if args.save_per_chunk:
                    save_bars_to_database(database=database, bars=normalized_bars, logger=logger)
                    manifest.update_chunk(chunk, status="saved", saved_at=utc_now())
                else:
                    buffered_bars.extend(normalized_bars)

                coverage_after_save: HistoryCoverageSummary | None = None
                if args.verify_db and args.save_per_chunk:
                    coverage_after_save = verify_chunk_in_database(
                        database=database,
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        chunk=chunk,
                        interval_delta=interval_delta,
                        timezone_name=history_range.timezone_name,
                    )

                    if not coverage_after_save.is_complete and args.repair_missing and coverage_after_save.missing_ranges:
                        log_event(
                            logger,
                            logging.WARNING,
                            "history.chunk_repair_start",
                            "Chunk save verification found gaps; attempting automatic repair",
                            chunk_index=chunk.index,
                            missing_ranges=[
                                {
                                    "start": item.start,
                                    "end": item.end,
                                    "missing_count": item.missing_count,
                                }
                                for item in coverage_after_save.missing_ranges
                            ],
                        )

                        for missing_range in coverage_after_save.missing_ranges:
                            repair_chunk = ChunkPlan(
                                index=chunk.index,
                                start=missing_range.start,
                                end_exclusive=missing_range.end + interval_delta,
                                end_display=missing_range.end,
                                timezone_name=history_range.timezone_name,
                            )
                            repair_bars, repair_source, repair_attempts = query_chunk_with_retry(
                                args=args,
                                chunk=repair_chunk,
                                symbol=symbol,
                                exchange=exchange,
                                interval=interval,
                                interval_delta=interval_delta,
                                gateway_context=gateway_context,
                                logger=logger,
                            )
                            total_attempts += repair_attempts
                            source_used = combine_source_used(source_used, repair_source)
                            normalized_repair_bars, _repair_duplicates = normalize_chunk_bars(
                                repair_bars,
                                chunk=repair_chunk,
                                timezone_name=history_range.timezone_name,
                                interval_delta=interval_delta,
                            )
                            if normalized_repair_bars:
                                save_bars_to_database(
                                    database=database,
                                    bars=normalized_repair_bars,
                                    logger=logger,
                                )

                        coverage_after_save = verify_chunk_in_database(
                            database=database,
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            chunk=chunk,
                            interval_delta=interval_delta,
                            timezone_name=history_range.timezone_name,
                        )

                    if coverage_after_save.is_complete:
                        entry = manifest.update_chunk(
                            chunk,
                            status="verified",
                            attempts=total_attempts,
                            source_used=source_used,
                            bar_count=coverage_after_save.total_count,
                            first_dt=coverage_after_save.first_dt,
                            last_dt=coverage_after_save.last_dt,
                            last_error=None,
                            verified_at=utc_now(),
                        )
                    else:
                        log_coverage_summary(
                            logger,
                            "history.chunk_verify_failed",
                            "Chunk exists in sqlite but still has missing bars after save/repair",
                            coverage_after_save,
                        )
                        entry = manifest.update_chunk(
                            chunk,
                            status="failed",
                            attempts=total_attempts,
                            source_used=source_used,
                            bar_count=coverage_after_save.total_count,
                            first_dt=coverage_after_save.first_dt,
                            last_dt=coverage_after_save.last_dt,
                            last_error=(
                                f"sqlite coverage incomplete after save: "
                                f"missing_count={coverage_after_save.missing_count}, "
                                f"gap_count={coverage_after_save.gap_count}"
                            ),
                        )
                        if strict_mode:
                            chunk_results.append(chunk_result_from_manifest(entry))
                            continue
                else:
                    entry = manifest.update_chunk(
                        chunk,
                        status="saved" if args.save_per_chunk else "pending",
                        attempts=total_attempts,
                        source_used=source_used,
                        bar_count=saved_count,
                        first_dt=first_dt,
                        last_dt=last_dt,
                        last_error=None,
                    )

                chunk_results.append(chunk_result_from_manifest(entry))
                log_event(
                    logger,
                    logging.INFO,
                    "history.chunk_done",
                    "Completed one history chunk",
                    chunk_index=chunk.index,
                    chunk_start=chunk.start,
                    chunk_end_exclusive=chunk.end_exclusive,
                    status=entry["status"],
                    attempts=entry["attempts"],
                    source_used=entry.get("source_used"),
                    count=entry.get("bar_count"),
                    first_dt=entry.get("first_dt"),
                    last_dt=entry.get("last_dt"),
                )
            except (HistoryDownloadError, DatabaseSaveError, ConfigurationError) as exc:
                error_class = classify_history_error(exc)
                entry = manifest.update_chunk(
                    chunk,
                    status="failed",
                    attempts=total_attempts or int(entry.get("attempts", 0) or 0),
                    source_used=source_used,
                    bar_count=saved_count,
                    first_dt=first_dt,
                    last_dt=last_dt,
                    last_error=str(exc),
                )
                chunk_results.append(chunk_result_from_manifest(entry))
                log_event(
                    logger,
                    logging.ERROR,
                    "history.chunk_failed",
                    "History chunk failed",
                    chunk_index=chunk.index,
                    chunk_start=chunk.start,
                    chunk_end_exclusive=chunk.end_exclusive,
                    error_class=error_class,
                    error_message=str(exc),
                )

        if buffered_bars:
            full_chunk = ChunkPlan(
                index=0,
                start=history_range.start,
                end_exclusive=history_range.end_exclusive,
                end_display=history_range.end_display,
                timezone_name=history_range.timezone_name,
            )
            buffered_bars, _duplicate_count = normalize_chunk_bars(
                buffered_bars,
                chunk=full_chunk,
                timezone_name=history_range.timezone_name,
                interval_delta=interval_delta,
            )
            save_bars_to_database(database=database, bars=buffered_bars, logger=logger)

        final_coverage = analyze_history_coverage(
            bars=load_final_bars(
                database=database,
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                history_range=history_range,
                logger=logger,
            ),
            history_range=history_range,
        )
        log_coverage_summary(
            logger,
            "history.final_coverage",
            "Final sqlite coverage after download",
            final_coverage,
        )

        csv_path: Path | None = None
        final_bars: list[BarData] = []
        if args.csv_copy:
            final_bars = load_final_bars(
                database=database,
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                history_range=history_range,
                logger=logger,
            )
            csv_path = export_csv_copy(
                vt_symbol=args.vt_symbol,
                interval_value=args.interval,
                start=history_range.start,
                end=history_range.end_display,
                bars=final_bars,
            )
            log_event(
                logger,
                logging.INFO,
                "history.csv_exported",
                "Exported CSV copy from sqlite coverage",
                csv_path=csv_path,
                count=len(final_bars),
            )

        instrument_path: Path | None = None
        if gateway_context is not None:
            instrument_path = build_instrument_config_path(args.vt_symbol)
            contract_payload = extract_contract_payload(gateway_context.contract)
            save_contract_payload(contract_payload, instrument_path)
            log_event(
                logger,
                logging.INFO,
                "instrument.updated",
                "Updated local instrument metadata file",
                instrument_path=instrument_path,
                contract=contract_payload,
            )

        failed_chunks = [item for item in chunk_results if item.status == "failed"]
        summary_payload = {
            "vt_symbol": args.vt_symbol,
            "interval": args.interval,
            "timezone": history_range.timezone_name,
            "requested_start": history_range.start,
            "requested_end_exclusive": history_range.end_exclusive,
            "requested_end_display": history_range.end_display,
            "requested_start_utc": history_range.start_utc,
            "requested_end_exclusive_utc": history_range.end_exclusive_utc,
            "total_unique_count": final_coverage.total_count,
            "expected_count": final_coverage.expected_count,
            "missing_count_after_save": final_coverage.missing_count,
            "first_dt": final_coverage.first_dt,
            "last_dt": final_coverage.last_dt,
            "manifest_path": manifest_path,
            "source": args.source,
            "strict_completeness": strict_mode,
            "allow_partial": bool(args.allow_partial),
            "chunk_results": [
                {
                    "index": item.index,
                    "start": item.start,
                    "end_exclusive": item.end_exclusive,
                    "end_display": item.end_display,
                    "timezone": item.timezone,
                    "status": item.status,
                    "attempts": item.attempts,
                    "source_used": item.source_used,
                    "count": item.count,
                    "first_dt": item.first_dt,
                    "last_dt": item.last_dt,
                    "error_message": item.error_message,
                }
                for item in chunk_results
            ],
            "final_coverage": {
                "total_count": final_coverage.total_count,
                "first_dt": final_coverage.first_dt,
                "last_dt": final_coverage.last_dt,
                "expected_count": final_coverage.expected_count,
                "missing_count": final_coverage.missing_count,
                "gap_count": final_coverage.gap_count,
                "largest_gap": (
                    {
                        "start": final_coverage.largest_gap.start,
                        "end": final_coverage.largest_gap.end,
                        "missing_count": final_coverage.largest_gap.missing_count,
                    }
                    if final_coverage.largest_gap is not None
                    else None
                ),
                "missing_ranges": [
                    {"start": item.start, "end": item.end, "missing_count": item.missing_count}
                    for item in final_coverage.missing_ranges
                ],
            },
            "repair_command": build_repair_command(
                vt_symbol=args.vt_symbol,
                interval_value=args.interval,
                start=args.start,
                end=args.end,
                timezone_name=args.timezone,
                chunk_days=max(1, min(args.chunk_days, 3)),
                source="auto",
            ),
            "csv_path": csv_path,
            "instrument_path": instrument_path,
        }
        print_json_block("History download summary:", summary_payload)

        if strict_mode and (failed_chunks or not final_coverage.is_complete):
            raise HistoryDownloadError(
                f"历史下载结束，但数据库仍不完整: failed_chunks={len(failed_chunks)}, "
                f"missing_count={final_coverage.missing_count}, gap_count={final_coverage.gap_count}"
            )
        return 0
    except (ConfigurationError, ValueError) as exc:
        log_event(logger, logging.ERROR, "history.config_error", str(exc), env_file=ENV_FILE)
        return 1
    except GatewayConnectionError as exc:
        recent_logs = gateway_context.observer.recent_log_messages() if gateway_context else []
        log_event(
            logger,
            logging.ERROR,
            "history.connection_error",
            str(exc),
            recent_logs=recent_logs,
        )
        return 2
    except HistoryDownloadError as exc:
        recent_logs = gateway_context.observer.recent_log_messages() if gateway_context else []
        log_event(
            logger,
            logging.ERROR,
            "history.download_error",
            str(exc),
            recent_logs=recent_logs,
        )
        return 3 if strict_mode else 0
    except DatabaseSaveError as exc:
        log_event(logger, logging.ERROR, "history.database_error", str(exc))
        return 4
    except KeyboardInterrupt:
        log_event(logger, logging.WARNING, "history.interrupted", "Interrupted by user")
        return 130
    except Exception:
        logger.exception(
            "Unexpected error during OKX history download",
            extra={"event": "history.unexpected_error"},
        )
        return 5
    finally:
        if gateway_context is not None:
            try:
                gateway_context.main_engine.close()
            except Exception:
                logger.exception(
                    "Failed to close MainEngine cleanly",
                    extra={"event": "history.close_error"},
                )


if __name__ == "__main__":
    raise SystemExit(main())
