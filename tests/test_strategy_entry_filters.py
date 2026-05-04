from __future__ import annotations

import unittest
from datetime import datetime
from typing import Any

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from strategies.okx_adaptive_mhf_strategy import OkxAdaptiveMhfStrategy


class FakeCtaEngine:
    """Minimal CTA engine stub for entry-filter tests."""

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
        datetime=datetime(2025, 1, 1, 12, 0),
        interval=Interval.MINUTE,
        volume=1.0,
        turnover=close_price,
        open_interest=0.0,
        open_price=close_price,
        high_price=close_price + 1.0,
        low_price=close_price - 1.0,
        close_price=close_price,
    )


class StrategyEntryFiltersTest(unittest.TestCase):
    def make_strategy(self, **setting_overrides: Any) -> OkxAdaptiveMhfStrategy:
        engine = FakeCtaEngine()
        setting: dict[str, Any] = {
            "capital_per_strategy": 5000.0,
            "risk_per_trade": 0.0005,
            "fixed_size": 0.01,
            "max_leverage": 0.5,
            "max_notional_ratio": 0.5,
            "max_volume": 0.0,
            "min_stop_pct": 0.0015,
            "max_trades_per_day": 10,
            "daily_loss_limit_pct": 0.02,
            "disable_after_daily_loss": True,
            "disable_after_bankrupt_guard": True,
            "min_atr_pct_for_entry": 0.0,
            "max_atr_pct_for_entry": 0.0,
            "require_regime_persistence_bars": 0,
            "min_breakout_atr": 0.0,
            "min_bars_between_entries": 0,
            "contract_size": 0.01,
            "min_volume": 0.01,
            "pricetick": 0.1,
            "rsi_long": 55.0,
            "rsi_short": 45.0,
        }
        setting.update(setting_overrides)
        strategy = OkxAdaptiveMhfStrategy(
            cta_engine=engine,
            strategy_name="entry_filter_test",
            vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
            setting=setting,
        )
        strategy.am_5m.inited = True
        strategy.regime = "long"
        strategy.regime_persistence_bars = 5
        strategy.breakout_high = 99.0
        strategy.breakout_low = 90.0
        strategy.rsi_1m_value = 60.0
        strategy.bar_index_1m = 100
        return strategy

    def test_min_atr_pct_for_entry_filters_low_volatility(self) -> None:
        strategy = self.make_strategy(min_atr_pct_for_entry=0.01)
        strategy.atr_1m_value = 0.5

        signal = strategy.generate_entry_signal(make_bar(100.0))

        self.assertEqual(signal, 0)

    def test_max_atr_pct_for_entry_filters_high_volatility(self) -> None:
        strategy = self.make_strategy(max_atr_pct_for_entry=0.01)
        strategy.atr_1m_value = 2.0

        signal = strategy.generate_entry_signal(make_bar(100.0))

        self.assertEqual(signal, 0)

    def test_require_regime_persistence_bars_blocks_early_entry(self) -> None:
        strategy = self.make_strategy(require_regime_persistence_bars=3)
        strategy.regime_persistence_bars = 1
        strategy.atr_1m_value = 1.0

        signal = strategy.generate_entry_signal(make_bar(100.0))

        self.assertEqual(signal, 0)

    def test_min_breakout_atr_blocks_weak_breakout(self) -> None:
        strategy = self.make_strategy(min_breakout_atr=2.0)
        strategy.atr_1m_value = 1.0

        signal = strategy.generate_entry_signal(make_bar(100.2))

        self.assertEqual(signal, 0)

    def test_min_bars_between_entries_blocks_reentry(self) -> None:
        strategy = self.make_strategy(min_bars_between_entries=5)
        strategy.atr_1m_value = 1.0
        strategy.last_entry_bar_index = 98
        strategy.bar_index_1m = 100

        signal = strategy.generate_entry_signal(make_bar(100.5))

        self.assertEqual(signal, 0)


if __name__ == "__main__":
    unittest.main()
