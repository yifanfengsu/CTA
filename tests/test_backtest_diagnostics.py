from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest_okx_mhf as backtest_mod


class FakeStatisticsEngine:
    """Simple fake engine for statistics fallback tests."""

    def __init__(self, should_raise: bool = False) -> None:
        self.should_raise = should_raise

    def calculate_statistics(self, df: pd.DataFrame, output: bool = False) -> dict[str, float]:
        if self.should_raise:
            raise RuntimeError("boom")
        return {"total_trade_count": 3, "total_net_pnl": 12.5}


class BacktestDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_backtest_diagnostics")
        self.logger.handlers.clear()

    def test_analyze_daily_pnl_detects_bankrupt(self) -> None:
        daily_df = pd.DataFrame(
            {
                "net_pnl": [-2000.0, -3500.0, 100.0],
                "commission": [10.0, 20.0, 5.0],
                "slippage": [5.0, 5.0, 2.0],
                "turnover": [1000.0, 1500.0, 800.0],
                "trade_count": [2, 3, 1],
            },
            index=pd.Index(
                [
                    pd.Timestamp("2025-01-01"),
                    pd.Timestamp("2025-01-02"),
                    pd.Timestamp("2025-01-03"),
                ],
                name="date",
            ),
        )

        diagnostics = backtest_mod.analyze_daily_pnl(daily_df, initial_capital=5000.0)

        self.assertTrue(diagnostics["bankrupt"])
        self.assertEqual(diagnostics["first_bankrupt_date"], "2025-01-02T00:00:00")
        self.assertEqual(diagnostics["final_balance"], -400.0)
        self.assertEqual(diagnostics["min_balance"], -500.0)
        self.assertEqual(diagnostics["total_net_pnl_sum"], -5400.0)
        self.assertEqual(diagnostics["daily_trade_count_sum"], 6.0)

    def test_calculate_statistics_safely_handles_exception(self) -> None:
        daily_df = pd.DataFrame({"net_pnl": [1.0]}, index=pd.Index([pd.Timestamp("2025-01-01")], name="date"))

        statistics, statistics_error = backtest_mod.calculate_statistics_safely(
            FakeStatisticsEngine(should_raise=True),
            daily_df,
            self.logger,
        )

        self.assertEqual(statistics, {})
        self.assertIn("RuntimeError", statistics_error or "")

    def test_build_stats_payload_splits_trade_counts(self) -> None:
        diagnostics = {
            "bankrupt": False,
            "first_bankrupt_date": None,
            "final_balance": 5200.0,
            "min_balance": 4700.0,
            "max_balance": 5300.0,
        }
        round_trip_stats = backtest_mod.RoundTripStats(
            closed_trade_count=11796,
            win_rate=55.0,
            profit_loss_ratio=1.2,
            average_win=15.0,
            average_loss=12.5,
            gross_profit=10000.0,
            gross_loss=8000.0,
        )

        payload = backtest_mod.build_stats_payload(
            statistics={"total_trade_count": 0, "total_net_pnl": 123.0},
            diagnostics=diagnostics,
            round_trip_stats=round_trip_stats,
            engine_trade_count=23592,
            order_count=12000,
            statistics_error=None,
        )

        self.assertEqual(payload["engine_trade_count"], 23592)
        self.assertEqual(payload["closed_round_trip_count"], 11796)
        self.assertEqual(payload["order_count"], 12000)
        self.assertEqual(payload["statistics_total_trade_count"], 0)
        self.assertNotIn("trade_count", payload)

    def test_load_setting_overrides_merges_file_and_cli(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            handle.write('{"fixed_size": 0.02, "risk_per_trade": 0.002}')
            file_path = Path(handle.name)

        try:
            overrides, resolved_path = backtest_mod.load_setting_overrides(
                str(file_path),
                '{"fixed_size": 0.01, "cooldown_bars": 10}',
            )
        finally:
            file_path.unlink(missing_ok=True)

        self.assertEqual(resolved_path, file_path)
        self.assertEqual(
            overrides,
            {"fixed_size": 0.01, "risk_per_trade": 0.002, "cooldown_bars": 10},
        )

        merged_setting = backtest_mod.fill_strategy_setting(
            raw_setting={"fixed_size": 0.0, "risk_per_trade": 0.003, "capital_per_strategy": 5000.0},
            instrument_meta=backtest_mod.InstrumentMeta(
                vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
                symbol="BTCUSDT_SWAP_OKX",
                exchange="GLOBAL",
                name="BTC-USDT-SWAP",
                size=0.01,
                pricetick=0.1,
                min_volume=0.01,
            ),
            capital=5000.0,
            logger=self.logger,
            setting_overrides=overrides,
        )
        self.assertEqual(merged_setting["fixed_size"], 0.01)
        self.assertEqual(merged_setting["risk_per_trade"], 0.002)
        self.assertEqual(merged_setting["cooldown_bars"], 10)

    def test_strategy_config_custom_path_loads_sanity_file(self) -> None:
        config_path = backtest_mod.resolve_path_arg(
            "config/strategy_sanity_min_size.json",
            backtest_mod.STRATEGY_CONFIG_PATH,
        )
        config_payload = backtest_mod.load_json_file(config_path)

        self.assertEqual(config_payload["class_name"], "OkxAdaptiveMhfStrategy")
        self.assertEqual(config_payload["setting"]["fixed_size"], 0.01)
        self.assertEqual(config_payload["setting"]["risk_per_trade"], 0.0005)


if __name__ == "__main__":
    unittest.main()
