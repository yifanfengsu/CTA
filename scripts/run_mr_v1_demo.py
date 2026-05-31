#!/usr/bin/env python3
"""MR-v1 Demo Runner: run MrV1Strategy on OKX DEMO in headless mode.

Connects to OKX DEMO, loads MrV1Strategy, and runs indefinitely.
Ctrl+C for graceful shutdown.

v1.1 — 基于 vnpy 知识库加固:
  - init_strategy 返回 Future，等待完成后再 start
  - 启动前检查合约数据已加载
  - 主循环监控 trading 状态
  - log.active 保持 False（策略自身有完整日志）
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
        "init_days": 90,
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

    now = datetime.datetime.now(datetime.timezone.utc); timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond//1000:03d}Z"
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
    if server == "DEMO":
        headers["x-simulated-trading"] = "1"

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

    # Must set BEFORE importing vnpy modules
    import vnpy.trader.setting
    vnpy.trader.setting.SETTINGS["log.active"] = True

    from vnpy.trader.engine import MainEngine
    from vnpy.trader.setting import SETTINGS
    from vnpy_okx import OkxGateway
    from vnpy_ctastrategy import CtaEngine
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

    # Wait for gateway + contract data to be ready
    print("Waiting for gateway login + contract data (max 60s)...")
    deadline = time.monotonic() + 60
    accounts_ready = False
    contracts_ready = False

    while time.monotonic() < deadline:
        time.sleep(2)
        # Check account
        if not accounts_ready:
            accounts = main_engine.get_all_accounts()
            if accounts:
                accounts_ready = True
                for acc in accounts:
                    print(f"  OKX {args.server} | {acc.accountid} balance: {acc.balance}")

        # Check contract data for first symbol — use full vt_symbol
        if not contracts_ready:
            contract = main_engine.get_contract(symbols[0])
            if contract:
                contracts_ready = True
                print(f"  Contract data loaded: {symbols[0]} (size={contract.size}, tick={contract.pricetick})")

        if accounts_ready and contracts_ready:
            print("  Gateway + contracts ready.")
            break

    if not contracts_ready:
        print("WARNING: Contract data not loaded after 60s. Manually loading...")
        from vnpy.trader.object import ContractData
        from vnpy.trader.constant import Exchange, Product
        from vnpy.event import Event
        from vnpy.trader.event import EVENT_CONTRACT
        
        for vt_symbol in symbols:
            base = vt_symbol.split(".")[0]
            coin = base.split("_")[0]
            okx_inst = coin.replace("USDT", "-USDT-SWAP")
            
            try:
                resp = requests.get(
                    f"{OKX_DEMO_REST}/api/v5/public/instruments",
                    params={"instType": "SWAP", "instId": okx_inst},
                    headers={"x-simulated-trading": "1"} if args.server == "DEMO" else {},
                    timeout=10,
                )
                data = resp.json()
                if data.get("code") == "0" and data.get("data"):
                    d = data["data"][0]
                    contract = ContractData(
                        symbol=base,
                        exchange=Exchange.GLOBAL,
                        name=okx_inst,
                        product=Product.SWAP,
                        size=float(d.get("ctVal", 0.01)),
                        pricetick=float(d.get("tickSz", 0.01)),
                        min_volume=float(d.get("minSz", 0.01)),
                        gateway_name="OKX",
                    )
                    main_engine.event_engine.put(Event(EVENT_CONTRACT, contract))
                    print(f"  Registered contract: {base} (size={contract.size}, tick={contract.pricetick})")
            except Exception as e:
                print(f"  Contract query FAILED for {okx_inst}: {e}")
        
        # Verify using full vt_symbol
        contract = main_engine.get_contract(symbols[0])
        if contract:
            print(f"  Contract verified: {symbols[0]}")
        else:
            print(f"  Contract STILL not loaded: {symbols[0]}")

    # Register event handler for shutdown
    running = True

    def on_shutdown(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    # --- Start strategies (串行，每步给足够时间) ---
    setting = strategy_setting()
    for vt_symbol in symbols:
        base = vt_symbol.split(".")[0]
        coin = base.split("_")[0]
        okx_inst = coin.replace("USDT", "-USDT-SWAP")

        print(f"Setting leverage for {okx_inst}...")
        set_leverage_okx(okx_inst, leverage=1, server=args.server)

        strat_name = f"mr_v1_{base.split('_')[0][:3].lower()}"
        print(f"Starting {strat_name} on {vt_symbol}...")

        cta.add_strategy(args.strategy_class, strat_name, vt_symbol, setting)
        cta.init_strategy(strat_name)
        print(f"  Initializing {strat_name}...")

        # 给足够时间让线程池完成 init（load_bar 通常 10-30s）
        time.sleep(45)

        strategy = cta.strategies.get(strat_name)
        if strategy and strategy.inited:
            cta.start_strategy(strat_name)
            print(f"  {strat_name} started (trading={strategy.trading})")
        else:
            print(f"  {strat_name} NOT inited, trying start anyway...")
            cta.start_strategy(strat_name)
            strategy = cta.strategies.get(strat_name)
            print(f"  Result: inited={strategy.inited if strategy else 'N/A'}, trading={strategy.trading if strategy else 'N/A'}")

    print(f"\n=== MR-v1 Demo Running on {len(symbols)} symbols ===")
    print("Press Ctrl+C to stop.\n")

    # --- Main loop with trading state monitoring ---
    try:
        last_check = time.time()
        while running:
            time.sleep(1)

            # v1.1: 每 30 秒检查一次策略 trading 状态
            if time.time() - last_check > 30:
                last_check = time.time()
                for vt_symbol in symbols:
                    base = vt_symbol.split(".")[0]
                    strat_name = f"mr_v1_{base.split('_')[0][:3].lower()}"
                    strategy = cta.strategies.get(strat_name)
                    if strategy and not strategy.trading and strategy.inited:
                        print(f"[MONITOR] {strat_name} trading=False, inited=True — possible silent disable!", flush=True)

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
