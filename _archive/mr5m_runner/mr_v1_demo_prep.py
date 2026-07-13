#!/usr/bin/env python3
"""MR-v1 Demo Preparation: vnpy backtest + OKX DEMO connectivity check.

Steps:
  1. vnpy backtest with MrV1Strategy on BTC 4h bars
  2. (Optional) OKX DEMO login check
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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


def parse_args():
    p = argparse.ArgumentParser(description="MR-v1 Demo Prep")
    p.add_argument("--mode", choices=["backtest", "check-demo", "all"], default="backtest")
    p.add_argument("--vt-symbol", default="BTCUSDT_SWAP_OKX.GLOBAL")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-03-31")
    p.add_argument("--capital", type=float, default=5000)
    p.add_argument("--rate", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "mr_v1_demo_prep"))
    return p.parse_args()


def run_vnpy_backtest(args) -> int:
    """Run vnpy BacktestingEngine with MrV1Strategy on 4h bars."""
    from vnpy.trader.constant import Interval
    from vnpy_ctastrategy.backtesting import BacktestingEngine

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=args.vt_symbol,
        interval=Interval.MINUTE,  # Use MINUTE, strategy aggregates to 240min
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        rate=args.rate,
        slippage=args.slippage,
        size=1,         # USDT-margined: each point = $1 per contract
        pricetick=0.1,  # OKX BTC tick size
        capital=args.capital,
    )

    # Add strategy
    from strategies.mr_v1_strategy import MrV1Strategy
    setting = {
        "notional_per_trade": 500.0,
        "lookback": 8,
        "atr_window": 14,
        "atr_stop": 1.0,
        "max_hold": 60,
        "init_days": 60,
        "price_offset": 2,
    }
    engine.add_strategy(MrV1Strategy, setting)

    print(f"Loading historical data for {args.vt_symbol}...")
    engine.load_data()
    print("Running backtest...")
    engine.run_backtesting()
    print("Computing statistics...")
    df = engine.calculate_result()
    stats = engine.calculate_statistics(output=False)

    print("\n=== MR-v1 vnpy Backtest Results ===")
    for key in ["total_return", "sharpe_ratio", "max_drawdown", "total_net_pnl",
                 "total_trade_count", "win_rate", "profit_factor"]:
        if key in stats:
            val = stats[key]
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

    if df is not None:
        df.to_csv(output_dir / "vnpy_daily_results.csv")
    stats["timestamp"] = datetime.now().isoformat()
    (output_dir / "vnpy_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    print(f"\nResults saved to {output_dir}")
    return 0


def check_okx_demo(args) -> int:
    """Connect to OKX DEMO and verify account."""
    import os
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    from vnpy.trader.engine import MainEngine
    from vnpy.trader.setting import SETTINGS
    from vnpy_okx import OkxGateway

    SETTINGS["log.active"] = False

    main_engine = MainEngine()
    main_engine.add_gateway(OkxGateway)

    gateway_name = "OKX"
    setting = {
        "API Key": os.getenv("OKX_API_KEY", ""),
        "Secret Key": os.getenv("OKX_SECRET_KEY", ""),
        "Passphrase": os.getenv("OKX_PASSPHRASE", ""),
        "Server": os.getenv("OKX_SERVER", "DEMO"),
        "Proxy Host": os.getenv("OKX_PROXY_HOST", ""),
        "Proxy Port": int(os.getenv("OKX_PROXY_PORT", "0")),
        "Spread Trading": "False",
        "Margin Currency": "USDT",
    }

    if not setting["API Key"]:
        print("ERROR: OKX_API_KEY not set in .env")
        return 1

    print(f"Connecting to OKX {setting['Server']}...")
    main_engine.connect(setting, gateway_name)

    import time
    time.sleep(5)

    accounts = main_engine.get_all_accounts()
    positions = main_engine.get_all_positions()

    print(f"\nAccounts: {len(accounts)}")
    for acc in accounts:
        print(f"  {acc.gateway_name} | Balance: {acc.balance} {acc.currency}")

    print(f"\nPositions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.vt_symbol} | Dir: {pos.direction.value} | Vol: {pos.volume} | PnL: {pos.pnl}")

    main_engine.close()
    print("\nOKX DEMO check complete")
    return 0


def main():
    ensure_headless_runtime()
    # Ensure strategies are importable
    sys.path.insert(0, str(PROJECT_ROOT))
    args = parse_args()

    if args.mode == "backtest":
        return run_vnpy_backtest(args)
    elif args.mode == "check-demo":
        return check_okx_demo(args)
    else:
        rc1 = run_vnpy_backtest(args)
        rc2 = check_okx_demo(args)
        return rc1 or rc2


if __name__ == "__main__":
    raise SystemExit(main())
