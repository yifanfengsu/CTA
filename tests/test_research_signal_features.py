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

import research_signal_features as feature_mod


def make_bars(minutes: int = 220) -> pd.DataFrame:
    """Build deterministic 1m OHLCV bars."""

    start = pd.Timestamp("2025-01-01T00:00:00+08:00")
    records = []
    previous_close = 100.0
    for index in range(minutes):
        close = 100.0 + 0.03 * index + 0.0004 * index * index
        open_price = previous_close
        high = max(open_price, close) + 0.4
        low = min(open_price, close) - 0.35
        records.append(
            {
                "datetime": (start + pd.Timedelta(minutes=index)).isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100.0 + index,
            }
        )
        previous_close = close
    return pd.DataFrame(records)


def price_at_minute(minute: int) -> float:
    """Return the close formula used by make_bars."""

    return 100.0 + 0.03 * minute + 0.0004 * minute * minute


def write_signal_trace(path: Path, row_count: int = 8, include_optional: bool = True) -> None:
    """Write a feature-rich signal trace."""

    records = []
    for index in range(row_count):
        minute = 40 + index * 10
        price = price_at_minute(minute)
        breakout_atr = 0.2 + index * 0.15
        atr = 1.0 + index * 0.05
        record = {
            "signal_id": f"SIG-{index + 1}",
            "datetime": (pd.Timestamp("2025-01-01T00:00:00+08:00") + pd.Timedelta(minutes=minute)).isoformat(),
            "vt_symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
            "direction": "long" if index % 2 == 0 else "short",
            "action": "entry",
            "price": price,
            "close_1m": price,
            "donchian_high": price - breakout_atr,
            "donchian_low": price - breakout_atr - 8.0,
            "breakout_distance": breakout_atr,
            "breakout_distance_atr": breakout_atr / atr,
            "atr_1m": atr,
            "atr_pct": atr / price,
            "rsi": 45.0 + index,
            "fast_ema_5m": price + 0.2,
            "slow_ema_5m": price - 0.2,
            "ema_spread": 0.4,
            "ema_spread_pct": 0.4 / price,
            "regime": "trend" if index % 2 == 0 else "range",
            "hour": int((minute // 60) % 24),
            "weekday": 2,
            "is_weekend": False,
            "stop_price": price - atr,
            "take_profit_price": price + 2 * atr,
        }
        if not include_optional:
            for column in ["rsi", "ema_spread_pct", "atr_pct", "fast_ema_5m", "slow_ema_5m"]:
                record.pop(column, None)
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)


class ResearchSignalFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_research_signal_features")
        self.logger.handlers.clear()

    def run_feature_research(self, tmp_dir: str, include_optional: bool = True) -> dict[str, object]:
        report_dir = Path(tmp_dir)
        trace_path = report_dir / "signal_trace.csv"
        output_dir = report_dir / "signal_feature_research"
        write_signal_trace(trace_path, include_optional=include_optional)
        return feature_mod.run_research(
            report_dir=report_dir,
            signal_trace_path=trace_path,
            output_dir=output_dir,
            horizons=[15, 30, 60, 120],
            bins=4,
            min_count=1,
            selected_features=feature_mod.parse_feature_list(None),
            timezone_name="Asia/Shanghai",
            bars_from_db=False,
            data_check_strict=False,
            logger=self.logger,
            bars_df=make_bars(),
        )

    def test_minimal_signal_trace_generates_feature_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = self.run_feature_research(tmp_dir)
            dataset = pd.read_csv(Path(tmp_dir) / "signal_feature_research" / "feature_dataset.csv")

        self.assertEqual(int(summary["entry_count"]), 8)
        self.assertIn("breakout_distance_atr", dataset.columns)
        self.assertIn("future_return_60m", dataset.columns)
        self.assertIn("good_signal_60m", dataset.columns)

    def test_numeric_feature_bins_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.run_feature_research(tmp_dir)
            bins = pd.read_csv(Path(tmp_dir) / "signal_feature_research" / "feature_bins.csv")

        breakout_bins = bins[bins["feature"] == "breakout_distance_atr"]
        self.assertGreaterEqual(len(breakout_bins.index), 2)
        self.assertIn("median_future_return_60m", breakout_bins.columns)

    def test_categorical_feature_bins_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.run_feature_research(tmp_dir)
            categorical = pd.read_csv(Path(tmp_dir) / "signal_feature_research" / "categorical_feature_bins.csv")

        self.assertIn("direction", set(categorical["feature"]))
        self.assertIn("regime", set(categorical["feature"]))

    def test_spearman_ic_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.run_feature_research(tmp_dir)
            ic = pd.read_csv(Path(tmp_dir) / "signal_feature_research" / "feature_ic.csv")

        row = ic[(ic["feature"] == "breakout_distance_atr") & (ic["target"] == "future_return_60m")]
        self.assertFalse(row.empty)
        self.assertGreaterEqual(int(row.iloc[0]["count"]), 3)

    def test_missing_fields_warn_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = self.run_feature_research(tmp_dir, include_optional=False)

        warnings = summary["warnings"]
        self.assertTrue(any("rsi" in warning for warning in warnings))

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.run_feature_research(tmp_dir)
            output_dir = Path(tmp_dir) / "signal_feature_research"

            for filename in [
                "feature_dataset.csv",
                "feature_summary.json",
                "feature_ic.csv",
                "feature_bins.csv",
                "categorical_feature_bins.csv",
                "feature_report.md",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
