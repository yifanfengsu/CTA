#!/usr/bin/env python3
"""Inspect OKX gateway fields locally without connecting to the exchange."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from common_runtime import ensure_headless_runtime, log_event, print_json_block, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Inspect OKX gateway default settings without opening any network connection."
    )
    parser.add_argument(
        "--gateway-name",
        default="OKX",
        help="Gateway name to inspect. Default: OKX.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose structured logs.",
    )
    return parser.parse_args()


def build_output(
    gateway_name: str,
    default_setting: dict[str, Any] | None,
    gateway_names: list[str],
) -> dict[str, Any]:
    """Build a readable payload for CLI output."""

    return {
        "gateway_name": gateway_name,
        "default_setting": default_setting,
        "gateway_names": gateway_names,
    }


def main() -> int:
    """Run gateway field inspection."""

    args = parse_args()
    ensure_headless_runtime()
    logger = setup_logging("inspect_okx_gateway", verbose=args.verbose)

    main_engine = None
    try:
        from vnpy.event import EventEngine
        from vnpy.trader.engine import MainEngine
        from vnpy_okx import OkxGateway

        log_event(
            logger,
            logging.INFO,
            "inspect.start",
            "Starting local OKX gateway inspection",
            gateway_name=args.gateway_name,
        )

        event_engine = EventEngine()
        main_engine = MainEngine(event_engine)
        main_engine.add_gateway(OkxGateway)

        default_setting = main_engine.get_default_setting(args.gateway_name)
        gateway_names = main_engine.get_all_gateway_names()
        payload = build_output(args.gateway_name, default_setting, gateway_names)

        log_event(
            logger,
            logging.INFO,
            "inspect.success",
            "Gateway inspection completed",
            gateway_name=args.gateway_name,
            registered_gateways=gateway_names,
        )

        print_json_block("OKX gateway inspection result:", payload)
        return 0
    except Exception:
        logger.exception(
            "OKX gateway inspection failed",
            extra={"event": "inspect.error", "gateway_name": args.gateway_name},
        )
        return 1
    finally:
        if main_engine is not None:
            try:
                main_engine.close()
            except Exception:
                logger.exception(
                    "Failed to close MainEngine cleanly",
                    extra={"event": "inspect.close_error"},
                )


if __name__ == "__main__":
    raise SystemExit(main())
