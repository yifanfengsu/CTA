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

import analyze_signal_outcomes as signal_mod


def write_signal_trace(path: Path, direction: str = "long", price: float = 100.0) -> None:
    """Write one minimal entry signal trace row."""

    trace_df = pd.DataFrame(
        [
            {
                "signal_id": "SIG-1",
                "datetime": "2025-01-01T00:00:00+08:00",
                "vt_symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "direction": direction,
                "action": "entry",
                "price": price,
                "close_1m": price,
                "breakout_distance_atr": 0.5,
                "atr_1m": 1.0,
                "regime": direction,
                "hour": 0,
                "weekday": 2,
                "is_weekend": False,
                "stop_price": price - 2.0 if direction == "long" else price + 2.0,
                "take_profit_price": price + 4.0 if direction == "long" else price - 4.0,
            }
        ]
    )
    trace_df.to_csv(path, index=False)


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


class AnalyzeSignalOutcomesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_analyze_signal_outcomes")
        self.logger.handlers.clear()

    def test_minimal_signal_trace_can_be_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "signal_trace.csv"
            write_signal_trace(trace_path)

            trace_df = signal_mod.read_signal_trace(trace_path, "Asia/Shanghai")
            entries = signal_mod.prepare_entry_signals(trace_df, [])

        self.assertEqual(len(entries.index), 1)
        self.assertEqual(entries.iloc[0]["direction"], "long")
        self.assertEqual(float(entries.iloc[0]["price"]), 100.0)

    def test_long_mfe_mae_is_computed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            trace_path = report_dir / "signal_trace.csv"
            output_dir = report_dir / "signal_outcomes"
            write_signal_trace(trace_path, direction="long", price=100.0)
            bars_df = make_bars(
                [
                    ("2025-01-01T00:01:00+08:00", 102.0, 99.0, 101.0),
                    ("2025-01-01T00:02:00+08:00", 103.0, 98.0, 102.0),
                ]
            )

            signal_mod.run_analysis(
                report_dir=report_dir,
                signal_trace_path=trace_path,
                output_dir=output_dir,
                horizons=[2],
                timezone_name="Asia/Shanghai",
                bars_from_db=False,
                logger=self.logger,
                bars_df=bars_df,
            )
            outcome_df = pd.read_csv(output_dir / "signal_outcomes.csv")

        self.assertAlmostEqual(float(outcome_df.iloc[0]["future_return_2m"]), 0.02)
        self.assertAlmostEqual(float(outcome_df.iloc[0]["mfe_2m"]), 0.03)
        self.assertAlmostEqual(float(outcome_df.iloc[0]["mae_2m"]), 0.02)

    def test_short_mfe_mae_is_computed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            trace_path = report_dir / "signal_trace.csv"
            output_dir = report_dir / "signal_outcomes"
            write_signal_trace(trace_path, direction="short", price=100.0)
            bars_df = make_bars(
                [
                    ("2025-01-01T00:01:00+08:00", 101.0, 98.0, 99.0),
                    ("2025-01-01T00:02:00+08:00", 102.0, 96.0, 97.0),
                ]
            )

            signal_mod.run_analysis(
                report_dir=report_dir,
                signal_trace_path=trace_path,
                output_dir=output_dir,
                horizons=[2],
                timezone_name="Asia/Shanghai",
                bars_from_db=False,
                logger=self.logger,
                bars_df=bars_df,
            )
            outcome_df = pd.read_csv(output_dir / "signal_outcomes.csv")

        self.assertAlmostEqual(float(outcome_df.iloc[0]["future_return_2m"]), 0.03)
        self.assertAlmostEqual(float(outcome_df.iloc[0]["mfe_2m"]), 0.04)
        self.assertAlmostEqual(float(outcome_df.iloc[0]["mae_2m"]), 0.02)

    def test_missing_signal_trace_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            code = signal_mod.main(
                [
                    "--report-dir",
                    tmp_dir,
                    "--signal-trace",
                    str(Path(tmp_dir) / "missing_signal_trace.csv"),
                    "--no-bars-from-db",
                ]
            )

        self.assertNotEqual(code, 0)

    def test_horizon_exceeds_data_warns_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            trace_path = report_dir / "signal_trace.csv"
            output_dir = report_dir / "signal_outcomes"
            write_signal_trace(trace_path)
            bars_df = make_bars([("2025-01-01T00:01:00+08:00", 101.0, 99.0, 100.5)])

            summary = signal_mod.run_analysis(
                report_dir=report_dir,
                signal_trace_path=trace_path,
                output_dir=output_dir,
                horizons=[5],
                timezone_name="Asia/Shanghai",
                bars_from_db=False,
                logger=self.logger,
                bars_df=bars_df,
            )

        self.assertTrue(any("horizon 5m" in warning for warning in summary["warnings"]))

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            trace_path = report_dir / "signal_trace.csv"
            output_dir = report_dir / "signal_outcomes"
            write_signal_trace(trace_path)
            bars_df = make_bars(
                [
                    ("2025-01-01T00:01:00+08:00", 101.0, 99.0, 100.5),
                    ("2025-01-01T00:02:00+08:00", 102.0, 98.5, 101.0),
                ]
            )

            signal_mod.run_analysis(
                report_dir=report_dir,
                signal_trace_path=trace_path,
                output_dir=output_dir,
                horizons=[1, 2],
                timezone_name="Asia/Shanghai",
                bars_from_db=False,
                logger=self.logger,
                bars_df=bars_df,
            )

            expected_files = [
                "signal_outcomes.csv",
                "outcome_summary.json",
                "outcome_by_side.csv",
                "outcome_by_hour.csv",
                "outcome_by_weekday.csv",
                "outcome_by_regime.csv",
                "outcome_by_breakout_distance_bucket.csv",
                "outcome_report.md",
            ]
            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
