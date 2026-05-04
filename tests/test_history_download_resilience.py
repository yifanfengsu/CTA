from __future__ import annotations

import logging
import sys
import unittest
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_okx_history as mod

from history_time_utils import ChunkPlan
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData


TZ = ZoneInfo("Asia/Shanghai")


def make_bar(dt: datetime, price: float = 100.0) -> BarData:
    return BarData(
        gateway_name="TEST",
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


class DownloadHistoryResilienceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_history_download_resilience")
        self.logger.handlers.clear()
        self.interval_delta = timedelta(minutes=1)
        self.chunk = ChunkPlan(
            index=1,
            start=datetime(2025, 1, 1, 0, 0, tzinfo=TZ),
            end_exclusive=datetime(2025, 1, 2, 0, 0, tzinfo=TZ),
            end_display=datetime(2025, 1, 1, 23, 59, tzinfo=TZ),
            timezone_name="Asia/Shanghai",
        )

    def build_args(self, **overrides: object) -> Namespace:
        base = {
            "vt_symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
            "interval": "1m",
            "start": "2025-01-01",
            "end": "2025-01-03",
            "timezone": "Asia/Shanghai",
            "chunk_days": 1,
            "server": "DEMO",
            "csv_copy": False,
            "timeout": 1.0,
            "resume": True,
            "checkpoint_dir": "data/history_manifests",
            "source": "gateway",
            "save_per_chunk": True,
            "verify_db": True,
            "repair_missing": False,
            "max_retries": 2,
            "retry_base_delay": 0.0,
            "retry_max_delay": 0.0,
            "throttle_seconds": 0.0,
            "strict_completeness": True,
            "allow_partial": False,
            "dry_run": False,
            "verbose": False,
        }
        base.update(overrides)
        return Namespace(**base)

    def test_retry_success(self) -> None:
        bars = [make_bar(self.chunk.start)]
        main_engine = SimpleNamespace(query_history=MagicMock(side_effect=[TimeoutError("timeout"), bars]))
        gateway_context = mod.GatewayContext(
            main_engine=main_engine,
            observer=SimpleNamespace(recent_log_messages=lambda limit=30: []),
            contract=SimpleNamespace(name="BTC-USDT-SWAP"),
        )
        args = self.build_args(source="gateway")

        with patch.object(mod.time, "sleep", return_value=None):
            result_bars, source_used, attempts = mod.query_chunk_with_retry(
                args=args,
                chunk=self.chunk,
                symbol="BTCUSDT_SWAP_OKX",
                exchange=Exchange.GLOBAL,
                interval=Interval.MINUTE,
                interval_delta=self.interval_delta,
                gateway_context=gateway_context,
                logger=self.logger,
            )

        self.assertEqual(source_used, "gateway")
        self.assertEqual(attempts, 2)
        self.assertEqual([bar.datetime for bar in result_bars], [self.chunk.start])
        self.assertEqual(main_engine.query_history.call_count, 2)

    def test_download_filters_half_open_range(self) -> None:
        overlapping_bars = [
            make_bar(self.chunk.start),
            make_bar(self.chunk.end_display),
            make_bar(self.chunk.end_exclusive),
        ]
        gateway_context = mod.GatewayContext(
            main_engine=SimpleNamespace(query_history=MagicMock(return_value=overlapping_bars)),
            observer=SimpleNamespace(recent_log_messages=lambda limit=30: []),
            contract=SimpleNamespace(name="BTC-USDT-SWAP"),
        )

        result_bars = mod.query_gateway_chunk(
            gateway_context=gateway_context,
            symbol="BTCUSDT_SWAP_OKX",
            exchange=Exchange.GLOBAL,
            interval=Interval.MINUTE,
            chunk=self.chunk,
            timezone_name="Asia/Shanghai",
            interval_delta=self.interval_delta,
        )

        self.assertEqual([bar.datetime for bar in result_bars], [self.chunk.start, self.chunk.end_display])

    def test_download_does_not_miss_last_minute_when_gateway_end_exclusive(self) -> None:
        half_open_bars = [
            make_bar(self.chunk.start),
            make_bar(self.chunk.end_display),
        ]
        gateway_context = mod.GatewayContext(
            main_engine=SimpleNamespace(query_history=MagicMock(return_value=half_open_bars)),
            observer=SimpleNamespace(recent_log_messages=lambda limit=30: []),
            contract=SimpleNamespace(name="BTC-USDT-SWAP"),
        )

        result_bars = mod.query_gateway_chunk(
            gateway_context=gateway_context,
            symbol="BTCUSDT_SWAP_OKX",
            exchange=Exchange.GLOBAL,
            interval=Interval.MINUTE,
            chunk=self.chunk,
            timezone_name="Asia/Shanghai",
            interval_delta=self.interval_delta,
        )

        self.assertEqual(result_bars[-1].datetime, self.chunk.end_display)
        self.assertNotIn(self.chunk.end_exclusive, [bar.datetime for bar in result_bars])


if __name__ == "__main__":
    unittest.main()
