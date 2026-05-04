from __future__ import annotations

import logging
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest_okx_mhf as backtest_mod
import verify_okx_history as verify_mod

from history_time_utils import parse_history_range
from history_utils import HistoryCoverageSummary, MissingRange
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData


TZ = ZoneInfo("Asia/Shanghai")


def make_db_bar(dt: datetime, price: float = 100.0) -> BarData:
    return BarData(
        gateway_name="DB",
        symbol="BTCUSDT_SWAP_OKX",
        exchange=Exchange.GLOBAL,
        datetime=dt,
        interval=Interval.MINUTE,
        volume=1.0,
        turnover=price,
        open_interest=0.0,
        open_price=price,
        high_price=price + 1,
        low_price=price - 1,
        close_price=price,
    )


def build_three_day_bars() -> list[BarData]:
    start = datetime(2025, 1, 1, 0, 0, tzinfo=TZ)
    end_exclusive = datetime(2025, 1, 4, 0, 0, tzinfo=TZ)
    current = start
    bars: list[BarData] = []
    while current < end_exclusive:
        bars.append(make_db_bar(current))
        current += timedelta(minutes=1)
    return bars


class VerifyOkxHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_verify_okx_history")
        self.logger.handlers.clear()
        self.history_range = parse_history_range(
            start_arg="2025-01-01",
            end_arg="2025-01-03",
            interval_delta=timedelta(minutes=1),
            timezone_name="Asia/Shanghai",
        )

    def test_verify_detects_user_current_failure_shape(self) -> None:
        bars = [
            bar
            for bar in build_three_day_bars()
            if bar.datetime
            not in {
                datetime(2025, 1, 2, 7, 59, tzinfo=TZ),
                datetime(2025, 1, 3, 7, 59, tzinfo=TZ),
            }
            and bar.datetime >= datetime(2025, 1, 1, 8, 0, tzinfo=TZ)
        ]
        fake_db = SimpleNamespace(load_bar_data=MagicMock(return_value=bars))

        with patch("vnpy.trader.database.get_database", return_value=fake_db):
            result = verify_mod.verify_history_range(
                vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
                interval_value="1m",
                start_arg="2025-01-01",
                end_arg="2025-01-03",
                timezone_name="Asia/Shanghai",
            )

        self.assertEqual(result.coverage.expected_count, 4320)
        self.assertEqual(result.coverage.missing_count, 482)
        self.assertEqual(result.coverage.largest_gap, MissingRange(
            start=datetime(2025, 1, 1, 0, 0, tzinfo=TZ),
            end=datetime(2025, 1, 1, 7, 59, tzinfo=TZ),
            missing_count=480,
        ))
        self.assertEqual(
            result.coverage.missing_ranges[-2:],
            [
                MissingRange(
                    start=datetime(2025, 1, 2, 7, 59, tzinfo=TZ),
                    end=datetime(2025, 1, 2, 7, 59, tzinfo=TZ),
                    missing_count=1,
                ),
                MissingRange(
                    start=datetime(2025, 1, 3, 7, 59, tzinfo=TZ),
                    end=datetime(2025, 1, 3, 7, 59, tzinfo=TZ),
                    missing_count=1,
                ),
            ],
        )

    def test_verify_complete_three_days(self) -> None:
        fake_db = SimpleNamespace(load_bar_data=MagicMock(return_value=build_three_day_bars()))

        with patch("vnpy.trader.database.get_database", return_value=fake_db):
            result = verify_mod.verify_history_range(
                vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
                interval_value="1m",
                start_arg="2025-01-01",
                end_arg="2025-01-03",
                timezone_name="Asia/Shanghai",
            )

        self.assertTrue(result.is_complete)
        self.assertEqual(result.coverage.expected_count, 4320)
        self.assertEqual(result.coverage.missing_count, 0)
        self.assertEqual(result.coverage.gap_count, 0)

    def test_repair_command_contains_timezone(self) -> None:
        fake_db = SimpleNamespace(load_bar_data=MagicMock(return_value=[]))

        with patch("vnpy.trader.database.get_database", return_value=fake_db):
            result = verify_mod.verify_history_range(
                vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
                interval_value="1m",
                start_arg="2025-01-01",
                end_arg="2025-01-03",
                timezone_name="Asia/Shanghai",
            )

        self.assertIn("--timezone Asia/Shanghai", result.repair_command)
        self.assertIn("--start 2025-01-01", result.repair_command)
        self.assertIn("--end 2025-01-03", result.repair_command)
        self.assertNotIn("2025-01-01T00:00:00 ", result.repair_command)

    def test_backtest_preflight_blocks_load_data_and_surfaces_repair_command(self) -> None:
        repair_command = (
            "python scripts/download_okx_history.py --vt-symbol BTCUSDT_SWAP_OKX.GLOBAL "
            "--interval 1m --start 2025-01-01 --end 2025-01-03 "
            "--chunk-days 3 --timezone Asia/Shanghai --resume --repair-missing --source auto"
        )
        coverage = HistoryCoverageSummary(
            total_count=2,
            first_dt=datetime(2025, 1, 1, 0, 0, tzinfo=TZ),
            last_dt=datetime(2025, 1, 1, 0, 1, tzinfo=TZ),
            expected_count=4320,
            missing_count=4318,
            gap_count=1,
            largest_gap=MissingRange(
                start=datetime(2025, 1, 1, 0, 2, tzinfo=TZ),
                end=datetime(2025, 1, 3, 23, 59, tzinfo=TZ),
                missing_count=4318,
            ),
            missing_ranges=[
                MissingRange(
                    start=datetime(2025, 1, 1, 0, 2, tzinfo=TZ),
                    end=datetime(2025, 1, 3, 23, 59, tzinfo=TZ),
                    missing_count=4318,
                )
            ],
        )
        verify_result = verify_mod.VerificationResult(
            vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
            interval_value="1m",
            history_range=self.history_range,
            coverage=coverage,
            repair_command=repair_command,
        )
        namespace = SimpleNamespace(
            vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
            start="2025-01-01",
            end="2025-01-03",
            timezone="Asia/Shanghai",
            capital=5000.0,
            rate=0.0005,
            slippage_mode="ticks",
            slippage=2.0,
            mode="bar",
            output_dir=None,
            skip_data_check=False,
            data_check_strict=True,
            verbose=False,
        )
        fake_engine_cls = MagicMock(name="BacktestingEngine")

        with (
            patch.object(backtest_mod, "verify_history_range", return_value=verify_result),
            patch.object(backtest_mod, "print_json_block", return_value=None),
            patch("vnpy_ctastrategy.backtesting.BacktestingEngine", fake_engine_cls),
        ):
            with self.assertRaises(backtest_mod.BacktestError) as context:
                backtest_mod.run_data_preflight(namespace, self.history_range, self.logger)

        self.assertIn(repair_command, str(context.exception))
        fake_engine_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
