#!/usr/bin/env python3
"""Connect to OKX in headless mode, verify auth, and persist contract metadata."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING, Any

from dotenv import dotenv_values

from common_runtime import (
    PROJECT_ROOT,
    ensure_headless_runtime,
    log_event,
    mask_secret,
    normalize_text,
    print_json_block,
    setup_logging,
    to_jsonable,
)

if TYPE_CHECKING:
    from vnpy.event import Event
    from vnpy.trader.object import ContractData, LogData


ENV_FILE: Path = PROJECT_ROOT / ".env"
DEFAULT_GATEWAY_NAME: str = "OKX"
DEFAULT_VT_SYMBOL: str = "BTCUSDT_SWAP_OKX.GLOBAL"


class ConfigurationError(Exception):
    """Raised when local configuration is invalid."""


class ConnectionCheckError(Exception):
    """Raised when OKX connectivity validation fails."""


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
    """Resolved gateway setting field names for the local vnpy_okx version."""

    api_key: str
    secret_key: str
    passphrase: str
    server: str
    proxy_host: str | None
    proxy_port: str | None

    def as_dict(self) -> dict[str, str | None]:
        """Serialize field map for logs."""

        return {
            "api_key": self.api_key,
            "secret_key": self.secret_key,
            "passphrase": self.passphrase,
            "server": self.server,
            "proxy_host": self.proxy_host,
            "proxy_port": self.proxy_port,
        }


class GatewayObserver:
    """Observe vn.py log and contract events during the connection check."""

    def __init__(self, logger: logging.Logger, target_vt_symbol: str, verbose: bool) -> None:
        self.logger: logging.Logger = logger
        self.target_vt_symbol: str = target_vt_symbol
        self.verbose: bool = verbose
        self.contract_ready: ThreadingEvent = ThreadingEvent()
        self.private_login_ready: ThreadingEvent = ThreadingEvent()
        self.private_login_success: bool = False
        self.private_login_failed: bool = False
        self.contract: ContractData | None = None
        self.recent_logs: deque[dict[str, Any]] = deque(maxlen=200)

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
            self.private_login_ready.set()
        elif "private api login failed" in message_lower:
            self.private_login_failed = True
            self.private_login_ready.set()

        if self.verbose:
            log_event(
                self.logger,
                logging.DEBUG,
                "okx.gateway_log",
                "Received vn.py gateway log",
                gateway_name=item["gateway_name"],
                log_message=item["msg"],
            )

    def on_contract(self, event: "Event") -> None:
        """Capture the target contract event."""

        contract: ContractData = event.data
        if contract.vt_symbol != self.target_vt_symbol:
            return

        self.contract = contract
        self.contract_ready.set()
        log_event(
            self.logger,
            logging.INFO,
            "okx.contract_detected",
            "Target contract metadata received",
            vt_symbol=contract.vt_symbol,
            contract_name=contract.name,
            gateway_name=contract.gateway_name,
        )

    def recent_log_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the latest captured vn.py logs."""

        return list(self.recent_logs)[-limit:]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Check OKX connectivity without placing orders and save contract metadata."
    )
    parser.add_argument(
        "--vt-symbol",
        default=DEFAULT_VT_SYMBOL,
        help=f"Target vt_symbol to validate. Default: {DEFAULT_VT_SYMBOL}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for contract + private login checks.",
    )
    parser.add_argument(
        "--server",
        choices=("REAL", "DEMO"),
        help="Override OKX_SERVER from .env.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def read_env_config(server_override: str | None) -> EnvConfig:
    """Read OKX credentials and runtime settings from the project .env file."""

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
        missing_text = ", ".join(missing_fields)
        raise ConfigurationError(f".env 缺字段或为空: {missing_text}")

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
    """Convert default gateway values into a connect-ready value."""

    if isinstance(value, list):
        return value[0] if value else ""
    return value


def find_setting_key(default_setting: dict[str, Any], *tokens: str) -> str | None:
    """Find a gateway setting key by semantic tokens instead of exact field names."""

    for key in default_setting:
        normalized_key = normalize_text(key)
        if all(normalize_text(token) in normalized_key for token in tokens):
            return key
    return None


def resolve_gateway_field_map(default_setting: dict[str, Any]) -> GatewayFieldMap:
    """Resolve the local OKX gateway field names from default settings."""

    api_key = find_setting_key(default_setting, "api", "key")
    secret_key = find_setting_key(default_setting, "secret", "key")
    passphrase = find_setting_key(default_setting, "passphrase")
    server = find_setting_key(default_setting, "server")
    proxy_host = find_setting_key(default_setting, "proxy", "host")
    proxy_port = find_setting_key(default_setting, "proxy", "port")

    required_missing: list[str] = []
    if not api_key:
        required_missing.append("api_key")
    if not secret_key:
        required_missing.append("secret_key")
    if not passphrase:
        required_missing.append("passphrase")
    if not server:
        required_missing.append("server")

    if required_missing:
        missing_text = ", ".join(required_missing)
        raise ConfigurationError(
            f"无法从 OKX default_setting 动态解析必需字段: {missing_text}"
        )

    return GatewayFieldMap(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        server=server,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
    )


def validate_server_option(default_setting: dict[str, Any], field_map: GatewayFieldMap, server: str) -> None:
    """Validate server override against the locally installed gateway options."""

    server_default = default_setting.get(field_map.server)
    if isinstance(server_default, list) and server not in server_default:
        options = ", ".join(str(option) for option in server_default)
        raise ConfigurationError(
            f"Server 取值无效: {server}，当前本地 OKX 可选项为: {options}"
        )


def build_connection_setting(
    default_setting: dict[str, Any],
    env_config: EnvConfig,
    field_map: GatewayFieldMap,
) -> dict[str, Any]:
    """Assemble the connect setting dict using dynamically resolved gateway fields."""

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
    """Mask secret values before logging or printing the connect setting."""

    sanitized: dict[str, Any] = {}
    for key, value in setting.items():
        normalized_key = normalize_text(key)
        if isinstance(value, str) and any(token in normalized_key for token in ("apikey", "secret", "passphrase")):
            sanitized[key] = mask_secret(value)
        else:
            sanitized[key] = to_jsonable(value)
    return sanitized


def wait_for_ready_state(
    main_engine: Any,
    observer: GatewayObserver,
    vt_symbol: str,
    timeout: float,
    logger: logging.Logger,
) -> ContractData:
    """Wait until both target contract metadata and private login success are observed."""

    deadline: float = time.monotonic() + timeout
    next_progress_log: float = time.monotonic() + 5.0

    while time.monotonic() < deadline:
        contract = observer.contract or main_engine.get_contract(vt_symbol)
        if contract is not None:
            observer.contract = contract
            observer.contract_ready.set()

        if (
            observer.contract_ready.is_set()
            and observer.private_login_success
            and observer.contract is not None
        ):
            return observer.contract

        if observer.private_login_failed:
            raise ConnectionCheckError(
                "API Key / Secret / Passphrase 不正确，或 DEMO / REAL 不匹配，Private API 登录失败。"
            )

        now = time.monotonic()
        if now >= next_progress_log:
            log_event(
                logger,
                logging.INFO,
                "okx.waiting",
                "Waiting for contract metadata and private login confirmation",
                vt_symbol=vt_symbol,
                contract_ready=observer.contract_ready.is_set(),
                private_login_success=observer.private_login_success,
            )
            next_progress_log = now + 5.0

        time.sleep(0.5)

    raise ConnectionCheckError(infer_failure_reason(observer, vt_symbol))


def infer_failure_reason(observer: GatewayObserver, vt_symbol: str) -> str:
    """Infer the clearest failure reason from collected vn.py logs."""

    joined_logs: str = "\n".join(item["msg"] for item in observer.recent_logs)
    logs_lower: str = joined_logs.lower()

    if "proxy" in logs_lower and any(
        token in logs_lower for token in ("refused", "error", "failed", "timeout")
    ):
        return "代理配置错误，无法通过代理建立到 OKX 的连接。"

    if any(
        token in logs_lower
        for token in (
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
            "tempor",
            "name resolution",
            "max retries exceeded",
            "network is unreachable",
            "timed out",
            "ssl",
            "disconnected",
            "exception catched by rest api",
            "exception catched by public api",
            "exception catched by private api",
        )
    ):
        return "网络问题，或代理配置错误，导致无法完成 OKX 连通性校验。"

    if "contract data received" in logs_lower and observer.contract is None:
        return f"已连到 OKX，但在超时时间内没有拿到目标合约元数据: {vt_symbol}"

    if observer.contract is not None and not observer.private_login_success:
        return "已拿到目标合约元数据，但没有等到 Private API 登录成功，请检查 API 权限和 DEMO/REAL 配置。"

    return f"在超时时间内没有完成 OKX 连接校验，也没有拿到目标合约元数据: {vt_symbol}"


def extract_contract_payload(contract: ContractData) -> dict[str, Any]:
    """Extract portable contract metadata fields for local config persistence."""

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


def build_output_path(vt_symbol: str) -> Path:
    """Build the instrument metadata output path from vt_symbol."""

    symbol_part, separator, _exchange_part = vt_symbol.partition(".")
    if not separator or not symbol_part:
        raise ConfigurationError(f"vt_symbol 格式无效: {vt_symbol}")

    filename = f"{symbol_part.lower()}.json"
    return PROJECT_ROOT / "config" / "instruments" / filename


def save_contract_payload(payload: dict[str, Any], output_path: Path) -> None:
    """Persist contract metadata to the config/instruments directory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def main() -> int:
    """Run OKX connectivity validation in headless mode."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("check_okx_connection", verbose=args.verbose)

    main_engine = None
    observer: GatewayObserver | None = None

    try:
        from vnpy.event import EventEngine
        from vnpy.trader.engine import MainEngine
        from vnpy.trader.event import EVENT_CONTRACT, EVENT_LOG
        from vnpy_okx import OkxGateway

        log_event(
            logger,
            logging.INFO,
            "okx.check.start",
            "Starting OKX connection check",
            vt_symbol=args.vt_symbol,
            timeout=args.timeout,
            server_override=args.server,
            env_file=ENV_FILE,
        )

        event_engine = EventEngine()
        main_engine = MainEngine(event_engine)
        main_engine.add_gateway(OkxGateway)

        default_setting = main_engine.get_default_setting(DEFAULT_GATEWAY_NAME)
        if default_setting is None:
            raise ConfigurationError("无法读取 OKX default_setting。请确认 OkxGateway 已正确安装。")

        env_config = read_env_config(args.server)
        field_map = resolve_gateway_field_map(default_setting)
        connect_setting = build_connection_setting(default_setting, env_config, field_map)

        observer = GatewayObserver(logger, args.vt_symbol, args.verbose)
        event_engine.register(EVENT_LOG, observer.on_log)
        event_engine.register(EVENT_CONTRACT, observer.on_contract)

        log_event(
            logger,
            logging.INFO,
            "okx.check.field_map",
            "Resolved local OKX gateway field mapping",
            field_map=field_map.as_dict(),
        )
        log_event(
            logger,
            logging.INFO,
            "okx.check.connect",
            "Connecting to OKX gateway",
            vt_symbol=args.vt_symbol,
            server=env_config.server,
            proxy_host=env_config.proxy_host or None,
            proxy_port=env_config.proxy_port,
            api_key=mask_secret(env_config.api_key),
            connect_setting=sanitize_setting(connect_setting),
        )

        main_engine.connect(connect_setting, DEFAULT_GATEWAY_NAME)
        contract = wait_for_ready_state(
            main_engine=main_engine,
            observer=observer,
            vt_symbol=args.vt_symbol,
            timeout=args.timeout,
            logger=logger,
        )

        payload = extract_contract_payload(contract)
        output_path = build_output_path(args.vt_symbol)
        save_contract_payload(payload, output_path)

        log_event(
            logger,
            logging.INFO,
            "okx.check.success",
            "OKX connection check passed",
            output_path=output_path,
            contract=payload,
        )

        print_json_block("Resolved OKX connect setting (sanitized):", sanitize_setting(connect_setting))
        print_json_block("Resolved contract metadata:", payload)
        print_json_block("Saved instrument file:", {"path": output_path})
        return 0
    except ConfigurationError as exc:
        log_event(
            logger,
            logging.ERROR,
            "okx.check.config_error",
            str(exc),
            env_file=ENV_FILE,
        )
        return 1
    except ConnectionCheckError as exc:
        recent_logs = observer.recent_log_messages() if observer is not None else []
        log_event(
            logger,
            logging.ERROR,
            "okx.check.connection_error",
            str(exc),
            recent_logs=recent_logs,
        )
        return 2
    except KeyboardInterrupt:
        log_event(
            logger,
            logging.WARNING,
            "okx.check.interrupted",
            "Interrupted by user",
        )
        return 130
    except Exception:
        logger.exception(
            "Unexpected error during OKX connection check",
            extra={"event": "okx.check.unexpected_error"},
        )
        return 3
    finally:
        if main_engine is not None:
            try:
                main_engine.close()
            except Exception:
                logger.exception(
                    "Failed to close MainEngine cleanly",
                    extra={"event": "okx.check.close_error"},
                )


if __name__ == "__main__":
    raise SystemExit(main())
