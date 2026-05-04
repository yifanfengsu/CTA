from __future__ import annotations

import unittest
from datetime import datetime
from typing import Any

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from strategies.okx_adaptive_mhf_strategy import OkxAdaptiveMhfStrategy


class FakeCtaEngine:
    """Minimal CTA engine stub for unit tests."""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.main_engine: Any | None = None

    def write_log(self, msg: str, strategy: Any) -> None:
        self.logs.append(msg)

    def send_order(self, strategy: Any, direction: Any, offset: Any, price: float, volume: float, stop: bool, lock: bool, net: bool) -> list[str]:
        return ["ORDER-1"]

    def cancel_all(self, strategy: Any) -> None:
        return None

    def put_strategy_event(self, strategy: Any) -> None:
        return None

    def load_bar(self, vt_symbol: str, days: int, interval: Interval, callback: Any, use_database: bool) -> list[BarData]:
        return []

    def get_pricetick(self, strategy: Any) -> float:
        return float(strategy.pricetick)

    def get_size(self, strategy: Any) -> float:
        return float(strategy.contract_size)

    def get_engine_type(self) -> str:
        return "backtesting"


def make_bar(close_price: float) -> BarData:
    return BarData(
        gateway_name="TEST",
        symbol="BTCUSDT_SWAP_OKX",
        exchange=Exchange.GLOBAL,
        datetime=datetime(2025, 1, 1, 0, 0),
        interval=Interval.MINUTE,
        volume=1.0,
        turnover=close_price,
        open_interest=0.0,
        open_price=close_price,
        high_price=close_price + 1.0,
        low_price=close_price - 1.0,
        close_price=close_price,
    )


class StrategyRiskCapsTest(unittest.TestCase):
    def make_strategy(self, **setting_overrides: Any) -> OkxAdaptiveMhfStrategy:
        engine = FakeCtaEngine()
        setting: dict[str, Any] = {
            "capital_per_strategy": 5000.0,
            "risk_per_trade": 0.001,
            "fixed_size": 0.0,
            "max_leverage": 1.0,
            "max_notional_ratio": 1.0,
            "max_volume": 0.0,
            "min_stop_pct": 0.0015,
            "max_trades_per_day": 20,
            "daily_loss_limit_pct": 0.02,
            "disable_after_daily_loss": True,
            "disable_after_bankrupt_guard": True,
            "contract_size": 0.01,
            "min_volume": 0.1,
            "pricetick": 0.1,
            "breakout_window": 20,
            "rsi_long": 55.0,
            "rsi_short": 45.0,
        }
        setting.update(setting_overrides)
        strategy = OkxAdaptiveMhfStrategy(
            cta_engine=engine,
            strategy_name="test_strategy",
            vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
            setting=setting,
        )
        return strategy

    def test_fixed_size_is_limited_by_max_leverage(self) -> None:
        strategy = self.make_strategy(fixed_size=10.0, max_leverage=0.5, max_notional_ratio=0.5)

        volume = strategy.compute_order_volume(close_price=100000.0)

        self.assertEqual(volume, 2.5)

    def test_risk_sizing_low_atr_is_limited_by_max_notional(self) -> None:
        strategy = self.make_strategy(fixed_size=0.0, risk_per_trade=0.01, max_leverage=0.5, max_notional_ratio=0.5)
        strategy.atr_1m_value = 0.1
        strategy.stop_atr = 1.2

        volume = strategy.compute_order_volume(close_price=100000.0)

        self.assertEqual(volume, 2.5)

    def test_stop_distance_respects_min_stop_pct(self) -> None:
        strategy = self.make_strategy(min_stop_pct=0.02)
        strategy.atr_1m_value = 0.1
        strategy.stop_atr = 1.2

        stop_distance = strategy.compute_stop_distance(close_price=100.0)

        self.assertEqual(stop_distance, 2.0)

    def test_generate_entry_signal_stops_after_max_trades_per_day(self) -> None:
        strategy = self.make_strategy(max_trades_per_day=2)
        strategy.am_5m.inited = True
        strategy.regime = "long"
        strategy.breakout_high = 99.0
        strategy.breakout_low = 90.0
        strategy.rsi_1m_value = 60.0
        strategy.daily_trade_count = 2

        signal = strategy.generate_entry_signal(make_bar(100.0))

        self.assertEqual(signal, 0)


if __name__ == "__main__":
    unittest.main()
