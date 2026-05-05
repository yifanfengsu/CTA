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

import research_entry_policies as entry_mod


def make_bars(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """Build a bars DataFrame from datetime/high/low/close rows."""

    return pd.DataFrame(
        [
            {
                "datetime": dt,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1.0,
            }
            for dt, high, low, close in rows
        ]
    )


def make_virtual_entry(direction: str = "long", entry_price: float = 100.0) -> entry_mod.VirtualEntry:
    """Build one immediate virtual entry for direct bracket tests."""

    return entry_mod.VirtualEntry(
        policy_name="immediate_baseline",
        row_number=0,
        signal_id="SIG-1",
        signal_dt=pd.Timestamp("2025-01-01T00:00:00+08:00"),
        direction=direction,
        side=direction,
        hour=0,
        weekday=2,
        is_weekend=False,
        breakout_distance_atr=0.5,
        atr_1m=1.0,
        original_price=entry_price,
        entry_dt=pd.Timestamp("2025-01-01T00:00:00+08:00"),
        entry_price=entry_price,
        bracket_start_index=0,
        skipped=False,
        skip_reason="",
    )


def write_signal_trace(path: Path, rows: list[dict[str, object]]) -> None:
    """Write signal trace rows with stable required columns."""

    defaults = {
        "vt_symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
        "direction": "long",
        "action": "entry",
        "price": 100.0,
        "close_1m": 100.0,
        "donchian_high": 99.5,
        "donchian_low": 100.5,
        "breakout_distance": 0.5,
        "breakout_distance_atr": 0.5,
        "atr_1m": 1.0,
        "regime": "trend",
        "hour": 0,
        "weekday": 2,
        "is_weekend": False,
    }
    records = []
    for index, row in enumerate(rows, start=1):
        record = dict(defaults)
        record.update(row)
        record.setdefault("signal_id", f"SIG-{index}")
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)


class ResearchEntryPoliciesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_research_entry_policies")
        self.logger.handlers.clear()

    def test_long_stop_and_tp_trigger(self) -> None:
        cases = [
            ("tp_first", [("2025-01-01T00:01:00+08:00", 102.2, 99.8, 102.0)], 2.0),
            ("stop_first", [("2025-01-01T00:01:00+08:00", 100.2, 98.9, 99.0)], -1.0),
        ]
        for expected_reason, rows, expected_r in cases:
            with self.subTest(expected_reason=expected_reason):
                bars = entry_mod.dataframe_bars_to_ohlc(make_bars(rows), "Asia/Shanghai")
                result = entry_mod.simulate_bracket(
                    bars,
                    pd.Series(bars["datetime"]),
                    make_virtual_entry("long"),
                    horizon_m=2,
                    stop_atr=1.0,
                    tp_atr=2.0,
                    warnings=[],
                )

                self.assertIsNotNone(result)
                self.assertEqual(result.exit_reason, expected_reason)
                self.assertAlmostEqual(result.r_multiple, expected_r)

    def test_short_stop_and_tp_trigger(self) -> None:
        cases = [
            ("tp_first", [("2025-01-01T00:01:00+08:00", 100.2, 97.8, 98.0)], 2.0),
            ("stop_first", [("2025-01-01T00:01:00+08:00", 101.1, 99.0, 101.0)], -1.0),
        ]
        for expected_reason, rows, expected_r in cases:
            with self.subTest(expected_reason=expected_reason):
                bars = entry_mod.dataframe_bars_to_ohlc(make_bars(rows), "Asia/Shanghai")
                result = entry_mod.simulate_bracket(
                    bars,
                    pd.Series(bars["datetime"]),
                    make_virtual_entry("short"),
                    horizon_m=2,
                    stop_atr=1.0,
                    tp_atr=2.0,
                    warnings=[],
                )

                self.assertIsNotNone(result)
                self.assertEqual(result.exit_reason, expected_reason)
                self.assertAlmostEqual(result.r_multiple, expected_r)

    def test_same_bar_stop_and_tp_is_stop_first(self) -> None:
        bars = entry_mod.dataframe_bars_to_ohlc(
            make_bars([("2025-01-01T00:01:00+08:00", 102.2, 98.8, 100.0)]),
            "Asia/Shanghai",
        )
        warnings: list[str] = []

        result = entry_mod.simulate_bracket(
            bars,
            pd.Series(bars["datetime"]),
            make_virtual_entry("long"),
            horizon_m=2,
            stop_atr=1.0,
            tp_atr=2.0,
            warnings=warnings,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.exit_reason, "stop_first")
        self.assertAlmostEqual(result.r_multiple, -1.0)
        self.assertTrue(any("stop first" in warning for warning in warnings))

    def test_skip_large_breakout_gt_1atr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            output_dir = report_dir / "entry_policy_research"
            trace_path = report_dir / "signal_trace.csv"
            write_signal_trace(
                trace_path,
                [
                    {"signal_id": "SIG-1", "datetime": "2025-01-01T00:00:00+08:00", "breakout_distance_atr": 0.5},
                    {"signal_id": "SIG-2", "datetime": "2025-01-01T00:03:00+08:00", "breakout_distance_atr": 1.5},
                ],
            )
            bars = make_bars(
                [
                    ("2025-01-01T00:01:00+08:00", 101.0, 99.5, 100.5),
                    ("2025-01-01T00:02:00+08:00", 101.5, 99.5, 101.0),
                    ("2025-01-01T00:03:00+08:00", 101.5, 99.5, 101.0),
                    ("2025-01-01T00:04:00+08:00", 101.0, 99.5, 100.5),
                    ("2025-01-01T00:05:00+08:00", 101.5, 99.5, 101.0),
                ]
            )

            entry_mod.run_research(
                report_dir=report_dir,
                signal_trace_path=trace_path,
                output_dir=output_dir,
                horizons=[2],
                stop_atr_grid=[1.0],
                tp_atr_grid=[2.0],
                max_wait_bars=3,
                timezone_name="Asia/Shanghai",
                bars_from_db=False,
                logger=self.logger,
                bars_df=bars,
            )
            leaderboard = pd.read_csv(output_dir / "entry_policy_leaderboard.csv")

        row = leaderboard[leaderboard["policy_name"] == "skip_large_breakout_gt_1atr"].iloc[0]
        self.assertEqual(int(row["entry_count"]), 1)
        self.assertEqual(int(row["skipped_count"]), 1)

    def test_delayed_confirm_1bar_enters_on_confirm_close(self) -> None:
        row = {
            "signal_id": "SIG-1",
            "datetime": "2025-01-01T00:00:00+08:00",
            "direction": "long",
            "price": 100.0,
            "close_1m": 100.0,
            "donchian_high": 99.5,
            "breakout_distance_atr": 0.5,
            "atr_1m": 1.0,
            "_signal_dt": pd.Timestamp("2025-01-01T00:00:00+08:00"),
            "hour": 0,
            "weekday": 2,
            "is_weekend": False,
        }
        bars = entry_mod.dataframe_bars_to_ohlc(
            make_bars(
                [
                    ("2025-01-01T00:01:00+08:00", 101.0, 99.8, 100.5),
                    ("2025-01-01T00:02:00+08:00", 102.0, 100.0, 101.5),
                ]
            ),
            "Asia/Shanghai",
        )

        decision = entry_mod.build_virtual_entry_for_policy(
            "delayed_confirm_1bar",
            0,
            row,
            bars,
            pd.Series(bars["datetime"]),
            max_wait_bars=3,
        )

        self.assertFalse(decision.skipped)
        self.assertEqual(decision.entry_dt, pd.Timestamp("2025-01-01T00:01:00+08:00"))
        self.assertAlmostEqual(float(decision.entry_price), 100.5)

    def test_pullback_policy_enters_on_breakout_level_touch(self) -> None:
        row = {
            "signal_id": "SIG-1",
            "datetime": "2025-01-01T00:00:00+08:00",
            "direction": "long",
            "price": 100.0,
            "close_1m": 100.0,
            "donchian_high": 99.5,
            "breakout_distance": 0.5,
            "breakout_distance_atr": 0.5,
            "atr_1m": 1.0,
            "_signal_dt": pd.Timestamp("2025-01-01T00:00:00+08:00"),
            "hour": 0,
            "weekday": 2,
            "is_weekend": False,
        }
        bars = entry_mod.dataframe_bars_to_ohlc(
            make_bars(
                [
                    ("2025-01-01T00:01:00+08:00", 100.2, 99.5, 99.8),
                    ("2025-01-01T00:02:00+08:00", 101.0, 99.6, 100.8),
                ]
            ),
            "Asia/Shanghai",
        )

        decision = entry_mod.build_virtual_entry_for_policy(
            "pullback_to_breakout_level_5bar",
            0,
            row,
            bars,
            pd.Series(bars["datetime"]),
            max_wait_bars=5,
        )

        self.assertFalse(decision.skipped)
        self.assertEqual(decision.entry_dt, pd.Timestamp("2025-01-01T00:01:00+08:00"))
        self.assertAlmostEqual(float(decision.entry_price), 99.5)

    def test_output_csv_json_md_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            output_dir = report_dir / "entry_policy_research"
            trace_path = report_dir / "signal_trace.csv"
            write_signal_trace(
                trace_path,
                [
                    {"signal_id": "SIG-1", "datetime": "2025-01-01T00:00:00+08:00", "direction": "long"},
                ],
            )
            bars = make_bars(
                [
                    ("2025-01-01T00:01:00+08:00", 101.0, 99.5, 100.5),
                    ("2025-01-01T00:02:00+08:00", 102.1, 100.0, 102.0),
                    ("2025-01-01T00:03:00+08:00", 102.5, 101.0, 102.2),
                    ("2025-01-01T00:04:00+08:00", 102.5, 101.0, 102.2),
                ]
            )

            entry_mod.run_research(
                report_dir=report_dir,
                signal_trace_path=trace_path,
                output_dir=output_dir,
                horizons=[2],
                stop_atr_grid=[1.0],
                tp_atr_grid=[2.0],
                max_wait_bars=3,
                timezone_name="Asia/Shanghai",
                bars_from_db=False,
                logger=self.logger,
                bars_df=bars,
            )

            expected_files = [
                "entry_policy_summary.json",
                "entry_policy_leaderboard.csv",
                "bracket_grid.csv",
                "policy_by_side.csv",
                "policy_by_hour.csv",
                "policy_report.md",
            ]
            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
