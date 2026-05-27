#!/usr/bin/env python3
"""MR-v1 Demo Runner: run MrV1Strategy on OKX DEMO in headless mode.

Connects to OKX DEMO, loads MrV1Strategy, and runs indefinitely.
Ctrl+C for graceful shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from common_runtime import PROJECT_ROOT, ensure_headless_runtime


OKX_DEMO_REST = "https://www.okx.com"


def parse_args():
    p = argparse.ArgumentParser(description="MR-v1 Demo Runner")
    p.add_argument("--symbols", default="BTCUSDT_SWAP_OKX.GLOBAL,ETHUSDT_SWAP_OKX.GLOBAL,SOLUSDT_SWAP_OKX.GLOBAL,LINKUSDT_SWAP_OKX.GLOBAL,DOGEUSDT_SWAP_OKX.GLOBAL")
    p.add_argument("--server", default="DEMO")
    p.add_argument("--strategy-class", default="MrV1Strategy")
    p.add_argument("--strategy-module", default="strategies.mr_v1_strategy")
    p.add_argument("--init-days", type=int, default=60)
    return p.parse_args()


def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    required = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: missing env vars: {missing}")
        print("Fill in .env file and retry.")
        sys.exit(1)


def build_gateway_setting(server: str) -> dict:
    return {
        "API Key": os.getenv("OKX_API_KEY", ""),
        "Secret Key": os.getenv("OKX_SECRET_KEY", ""),
        "Passphrase": os.getenv("OKX_PASSPHRASE", ""),
        "Server": server,
        "Proxy Host": os.getenv("OKX_PROXY_HOST", ""),
        "Proxy Port": int(os.getenv("OKX_PROXY_PORT", "0")),
        "Spread Trading": "False",
        "Margin Currency": "USDT",
    }


def strategy_setting() -> dict:
    return {
        "notional_per_trade": 500.0,
        "lookback": 8,
        "atr_window": 14,
        "atr_stop": 1.0,
        "max_hold": 60,
        "init_days": 60,
        "price_offset": 2,
    }


def set_leverage_okx(symbol: str, leverage: int = 1, server: str = "DEMO") -> bool:
    """Set leverage for a perpetual swap on OKX via REST API.

    symbol: e.g. "BTC-USDT-SWAP"
    """
    import base64
    import hmac
    import datetime

    api_key = os.getenv("OKX_API_KEY", "")
    secret = os.getenv("OKX_SECRET_KEY", "")
    passphrase = os.getenv("OKX_PASSPHRASE", "")

    if not all([api_key, secret, passphrase]):
        print(f"  SKIP leverage: missing API credentials")
        return False

    base_url = OKX_DEMO_REST
    if server != "DEMO":
        base_url = "https://www.okx.com"  # REAL — verify with user

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat("T", "milliseconds") + "Z"
    method = "POST"
    path = "/api/v5/account/set-leverage"
    body = json.dumps({
        "instId": symbol,
        "lever": str(leverage),
        "mgnMode": "cross",
    })

    sign_str = timestamp + method + path + body
    sign = base64.b64encode(
        hmac.new(secret.encode(), sign_str.encode(), "sha256").digest()
    ).decode()

    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(base_url + path, data=body, headers=headers, timeout=10)
        result = resp.json()
        if result.get("code") == "0":
            print(f"  Leverage set: {symbol} → {leverage}x")
            return True
        else:
            print(f"  Leverage FAILED {symbol}: {result.get('msg', 'unknown')}")
            return False
    except Exception as e:
        print(f"  Leverage ERROR {symbol}: {e}")
        return False


def main():
    ensure_headless_runtime()
    args = parse_args()
    load_env()

    sys.path.insert(0, str(PROJECT_ROOT))

    from vnpy.trader.engine import MainEngine
    from vnpy.trader.setting import SETTINGS
    from vnpy_okx import OkxGateway
    from vnpy_ctastrategy import CtaEngine

    SETTINGS["log.active"] = False
    symbols = [s.strip() for s in args.symbols.split(",")]

    # --- Engine setup ---
    main_engine = MainEngine()
    main_engine.add_gateway(OkxGateway)

    # CTA engine
    cta: CtaEngine = main_engine.add_engine(CtaEngine)
    cta.load_strategy_class_from_module(args.strategy_module)

    # --- Connect ---
    gateway_setting = build_gateway_setting(args.server)
    print(f"Connecting to OKX {args.server}...")
    main_engine.connect(gateway_setting, "OKX")

    # Wait for gateway to be ready
    print("Waiting for gateway login (max 30s)...")
    deadline = time.monotonic() + 30
    logged_in = False
    while time.monotonic() < deadline:
        time.sleep(1)
        accounts = main_engine.get_all_accounts()
        if accounts:
            logged_in = True
            for acc in accounts:
                print(f"  OKX {args.server} | {acc.accountid} balance: {acc.balance}")
            break

    if not logged_in:
        print("WARNING: No account data received. Continuing anyway...")

    # Register event handler for shutdown
    running = True

    def on_shutdown(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    # --- Start strategies ---
    setting = strategy_setting()
    for vt_symbol in symbols:
        # Set 1x leverage for safety
        # vt_symbol = "BTCUSDT_SWAP_OKX.GLOBAL" → inst_id = "BTC-USDT-SWAP"
        base = vt_symbol.split(".")[0]                     # "BTCUSDT_SWAP_OKX"
        coin = base.split("_")[0]                          # "BTCUSDT"
        okx_inst = coin.replace("USDT", "-USDT-SWAP")      # "BTC-USDT-SWAP"

        print(f"Setting leverage for {okx_inst}...")
        set_leverage_okx(okx_inst, leverage=1, server=args.server)

        strat_name = f"mr_v1_{base.split('_')[0][:3].lower()}"
        print(f"Starting {strat_name} on {vt_symbol}...")
        cta.add_strategy(args.strategy_class, strat_name, vt_symbol, setting)
        cta.init_strategy(strat_name)
        cta.start_strategy(strat_name)

    print(f"\n=== MR-v1 Demo Running on {len(symbols)} symbols ===")
    print("Press Ctrl+C to stop.\n")

    # --- Main loop ---
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # --- Shutdown ---
    print("Stopping strategies...")
    for vt_symbol in symbols:
        base = vt_symbol.split(".")[0]
        strat_name = f"mr_v1_{base.split('_')[0][:3].lower()}"
        try:
            cta.stop_strategy(strat_name)
        except Exception:
            pass

    print("Closing engine...")
    main_engine.close()
    print("Done.")


if __name__ == "__main__":
    main()
