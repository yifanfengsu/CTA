#!/usr/bin/env python3
"""MR-5m Demo Runner: run Mr5mStrategy on OKX DEMO in headless mode.

Based on the proven run_mr_v1_demo.py pattern.
Connects to OKX DEMO, loads Mr5mStrategy (5-min MR), and runs indefinitely.
"""

from __future__ import annotations

import argparse, base64, datetime, hmac, json, os, signal, sys, time
from pathlib import Path

import requests
from dotenv import load_dotenv
# 2026-07 重构批次6：脚本迁入 _archive/mr5m_runner/；共享依赖真身在
# scripts/（前向冻结区）与 core/data_io/，此处按新深度注入 sys.path。
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (
    str(_REPO_ROOT / "core" / "data_io"),
    str(_REPO_ROOT / "scripts"),
    str(_REPO_ROOT / "data_engineering" / "scripts"),
    *sorted(str(_q) for _q in (_REPO_ROOT / "research" / "_closed").glob("*/*/scripts")),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from common_runtime import PROJECT_ROOT, ensure_headless_runtime

OKX_DEMO_REST = "https://www.okx.com"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT_SWAP_OKX.GLOBAL,ETHUSDT_SWAP_OKX.GLOBAL,"
                   "SOLUSDT_SWAP_OKX.GLOBAL,LINKUSDT_SWAP_OKX.GLOBAL,DOGEUSDT_SWAP_OKX.GLOBAL")
    p.add_argument("--server", default="DEMO")
    p.add_argument("--strategy-class", default="Mr5mStrategy")
    p.add_argument("--strategy-module", default="strategies.mr_5m_strategy")
    return p.parse_args()


def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    required = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: missing env vars: {missing}")
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
        "lookback": 24,
        "atr_window": 14,
        "atr_stop": 1.0,
        "max_hold": 48,
        "init_days": 7,
        "price_offset": 2,
        "atr_filter_on": True,
        "atr_filter_threshold": 0.0,  # auto-detected
    }


def okx_inst_id(vt_symbol: str) -> str:
    coin = vt_symbol.split("_")[0]
    return coin.replace("USDT", "-USDT-SWAP")


def set_leverage_okx(symbol: str, leverage: int = 1, server: str = "DEMO") -> bool:
    api_key = os.getenv("OKX_API_KEY", "")
    secret = os.getenv("OKX_SECRET_KEY", "")
    passphrase = os.getenv("OKX_PASSPHRASE", "")
    if not all([api_key, secret, passphrase]):
        return False
    base_url = OKX_DEMO_REST
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond//1000:03d}Z"
    method, path = "POST", "/api/v5/account/set-leverage"
    body = json.dumps({"instId": symbol, "lever": str(leverage), "mgnMode": "cross"})
    sign_str = ts + method + path + body
    sign = base64.b64encode(hmac.new(secret.encode(), sign_str.encode(), "sha256").digest()).decode()
    headers = {
        "OK-ACCESS-KEY": api_key, "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
    }
    if server == "DEMO":
        headers["x-simulated-trading"] = "1"
    try:
        resp = requests.post(base_url + path, data=body, headers=headers, timeout=10)
        result = resp.json()
        ok = result.get("code") == "0"
        print(f"  {'OK' if ok else 'FAIL'}: {symbol} leverage={leverage}x  ({result.get('msg','')})")
        return ok
    except Exception as e:
        print(f"  ERROR: {symbol} {e}")
        return False


def main():
    ensure_headless_runtime()
    args = parse_args()
    load_env()
    sys.path.insert(0, str(PROJECT_ROOT))

    # Import vnpy modules
    import vnpy.trader.setting
    vnpy.trader.setting.SETTINGS["log.active"] = True
    from vnpy.trader.engine import MainEngine
    from vnpy_okx import OkxGateway
    from vnpy_ctastrategy import CtaEngine

    symbols = [s.strip() for s in args.symbols.split(",")]
    print(f"MR-5m Demo | {len(symbols)} symbols | server={args.server}")

    # Engine setup
    main_engine = MainEngine()
    main_engine.add_gateway(OkxGateway)
    cta: CtaEngine = main_engine.add_engine(CtaEngine)
    cta.init_engine()
    cta.load_strategy_class_from_module(args.strategy_module)

    # Connect
    gateway_setting = build_gateway_setting(args.server)
    print(f"Connecting to OKX {args.server}...")
    main_engine.connect(gateway_setting, "OKX")

    # Wait for gateway + contracts
    print("Waiting for gateway + contracts (max 60s)...")
    deadline = time.monotonic() + 60
    accounts_ready, contracts_ready = False, False
    while time.monotonic() < deadline:
        time.sleep(2)
        if not accounts_ready:
            accounts = main_engine.get_all_accounts()
            if accounts:
                accounts_ready = True
                for acc in accounts:
                    print(f"  Account: {acc.accountid} balance={acc.balance}")
        if not contracts_ready:
            contract = main_engine.get_contract(symbols[0])
            if contract:
                contracts_ready = True
                print(f"  Contract: {symbols[0]} (size={contract.size}, tick={contract.pricetick})")
        if accounts_ready and contracts_ready:
            print("  Ready.")
            break

    if not contracts_ready:
        print("  WARNING: Contract not loaded, fetching from REST API...")
        from vnpy.trader.object import ContractData
        from vnpy.trader.constant import Exchange, Product
        from vnpy.event import Event
        from vnpy.trader.event import EVENT_CONTRACT
        for vt_symbol in symbols:
            inst = okx_inst_id(vt_symbol)
            resp = requests.get(f"{OKX_DEMO_REST}/api/v5/public/instruments",
                params={"instType": "SWAP", "instId": inst},
                headers={"x-simulated-trading": "1"} if args.server == "DEMO" else {}, timeout=10)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                d = data["data"][0]
                contract = ContractData(
                    symbol=vt_symbol.split(".")[0], exchange=Exchange.GLOBAL, name=inst,
                    product=Product.SWAP, size=float(d.get("ctVal", 0.01)),
                    pricetick=float(d.get("tickSz", 0.01)),
                    min_volume=float(d.get("minSz", 0.01)), gateway_name="OKX",
                )
                main_engine.event_engine.put(Event(EVENT_CONTRACT, contract))
                print(f"  Registered: {vt_symbol}")

    # Set leverage
    for vt_symbol in symbols:
        set_leverage_okx(okx_inst_id(vt_symbol), leverage=1, server=args.server)

    # Start strategies (serial)
    setting = strategy_setting()
    for vt_symbol in symbols:
        base = vt_symbol.split(".")[0]
        strat_name = f"mr5m_{base.split('_')[0][:3].lower()}"
        print(f"Starting {strat_name}...")
        cta.add_strategy(args.strategy_class, strat_name, vt_symbol, setting)
        cta.init_strategy(strat_name)
        time.sleep(25)
        strategy = cta.strategies.get(strat_name)
        if strategy and strategy.inited:
            cta.start_strategy(strat_name)
            print(f"  {strat_name}: started (trading={strategy.trading})")
        else:
            print(f"  {strat_name}: init failed, trying start anyway")
            cta.start_strategy(strat_name)
            strategy = cta.strategies.get(strat_name)
            print(f"  Result: inited={strategy.inited if strategy else 'N/A'}")

    print(f"\n=== MR-5m Demo Running on {len(symbols)} symbols ===")
    print("Ctrl+C to stop.\n")

    # Main loop
    running = True

    def on_shutdown(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    last_check = time.time()
    try:
        while running:
            time.sleep(1)
            if time.time() - last_check > 30:
                last_check = time.time()
                for vt_symbol in symbols:
                    base = vt_symbol.split(".")[0]
                    sn = f"mr5m_{base.split('_')[0][:3].lower()}"
                    s = cta.strategies.get(sn)
                    if s and not s.trading and s.inited:
                        print(f"[MONITOR] {sn} trading=False — possible silent disable!", flush=True)
    except KeyboardInterrupt:
        pass

    # Shutdown
    print("Stopping strategies...")
    for vt_symbol in symbols:
        try:
            cta.stop_strategy(f"mr5m_{vt_symbol.split('.')[0].split('_')[0][:3].lower()}")
        except Exception:
            pass
    main_engine.close()
    print("Done.")


if __name__ == "__main__":
    main()
