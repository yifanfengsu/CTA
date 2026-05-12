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

import research_trend_opportunity_map as tom


SYMBOL = "AAAUSDT_SWAP_OKX.GLOBAL"


def make_closed_frame(close_values: list[float], freq: str = "4h") -> pd.DataFrame:
    datetimes = pd.date_range("2025-01-01T03:59:00+08:00", periods=len(close_values), freq=freq)
    rows = []
    previous = close_values[0]
    for index, close in enumerate(close_values):
        rows.append(
            {
                "datetime": datetimes[index],
                "open": previous,
                "high": max(previous, close) + 0.25,
                "low": min(previous, close) - 0.25,
                "close": close,
                "volume": 100.0 + index,
            }
        )
        previous = close
    return pd.DataFrame(rows)


def make_1m_bars(symbol: str, days: int, start: str = "2025-01-01T00:00:00+08:00") -> pd.DataFrame:
    rows = []
    start_dt = pd.Timestamp(start)
    previous = 100.0
    for index in range(days * 1440):
        close = 100.0 + index * 0.002
        rows.append(
            {
                "vt_symbol": symbol,
                "datetime": start_dt + pd.Timedelta(minutes=index),
                "open": previous,
                "high": max(previous, close) + 0.02,
                "low": min(previous, close) - 0.02,
                "close": close,
                "volume": 10.0 + index % 7,
            }
        )
        previous = close
    return pd.DataFrame(rows)


class ResearchTrendOpportunityMapTest(unittest.TestCase):
    def test_atr_move_segment_labeling(self) -> None:
        frame = make_closed_frame([100.0, 100.5, 101.5, 102.2, 103.4])
        frame["atr14"] = 1.0

        candidates = tom.atr_move_candidates(frame, SYMBOL, "4h", max_lookahead=4)
        labels = set().union(*(candidate["labels"] for candidate in candidates))

        self.assertIn("uptrend_2atr", labels)
        self.assertIn("uptrend_3atr", labels)

    def test_trend_efficiency_calculation(self) -> None:
        clean = tom.forward_trend_efficiency(pd.Series([100.0, 101.0, 102.0, 103.0]), 3)
        self.assertAlmostEqual(float(clean.iloc[0]), 1.0)

        choppy = tom.forward_trend_efficiency(pd.Series([100.0, 101.0, 100.0, 101.0]), 3)
        self.assertAlmostEqual(float(choppy.iloc[0]), 1.0 / 3.0)

    def test_run_length_calculation(self) -> None:
        frame = make_closed_frame([100.0, 101.0, 102.0, 103.0, 104.0, 103.5])

        candidates = tom.run_length_candidates(frame, SYMBOL, "4h")

        self.assertTrue(any(candidate["direction"] == "up" and candidate["end_idx"] == 4 for candidate in candidates))

    def test_trend_segment_merge_or_dedupe(self) -> None:
        frame = make_closed_frame([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        candidates = [
            {
                "symbol": SYMBOL,
                "timeframe": "4h",
                "direction": "up",
                "start_idx": 0,
                "end_idx": 3,
                "labels": {"uptrend_2atr"},
            },
            {
                "symbol": SYMBOL,
                "timeframe": "4h",
                "direction": "up",
                "start_idx": 2,
                "end_idx": 5,
                "labels": {"trend_efficiency_20_ge_0.45"},
            },
        ]

        merged = tom.merge_or_dedupe_segments(candidates, {(SYMBOL, "4h"): frame})

        self.assertEqual(len(merged.index), 1)
        self.assertIn("uptrend_2atr", merged.iloc[0]["labels"])
        self.assertIn("trend_efficiency_20_ge_0.45", merged.iloc[0]["labels"])

    def test_legacy_trade_aligns_to_trend_segment(self) -> None:
        segments = pd.DataFrame(
            [
                {
                    "trend_segment_id": "seg_1",
                    "symbol": SYMBOL,
                    "timeframe": "4h",
                    "start_time": "2025-01-01T03:59:00+08:00",
                    "end_time": "2025-01-02T03:59:00+08:00",
                    "start_idx": 0,
                    "end_idx": 6,
                    "duration_bars": 7,
                    "abs_trend_return": 0.10,
                    "is_major_trend": True,
                }
            ]
        )
        trades = pd.DataFrame(
            [
                {
                    "strategy_source": "unit",
                    "policy_or_group": "policy",
                    "symbol": SYMBOL,
                    "timeframe": "4h",
                    "trade_id": "t1",
                    "entry_time": "2025-01-01T08:00:00+08:00",
                    "exit_time": "2025-01-01T20:00:00+08:00",
                    "pnl": 1.0,
                }
            ]
        )

        coverage = tom.align_legacy_trades_to_segments(trades, segments, ["4h"])
        real = coverage[~coverage["is_synthetic_missed_segment"]].iloc[0]

        self.assertTrue(bool(real["entered_trend_segment"]))
        self.assertEqual(real["trend_segment_id"], "seg_1")

    def test_entry_phase_identification(self) -> None:
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        end = pd.Timestamp("2025-01-01T09:00:00Z")

        self.assertEqual(tom.entry_phase(pd.Timestamp("2025-01-01T01:00:00Z"), start, end), "early")
        self.assertEqual(tom.entry_phase(pd.Timestamp("2025-01-01T04:00:00Z"), start, end), "middle")
        self.assertEqual(tom.entry_phase(pd.Timestamp("2025-01-01T08:00:00Z"), start, end), "late")

    def test_pre_trend_feature_comparison_output(self) -> None:
        frame = tom.compute_context_indicators(make_closed_frame([100.0 + index * 0.2 for index in range(130)]))
        frame["market_breadth"] = 0.6
        frame["market_correlation"] = 0.3
        segments = pd.DataFrame(
            [
                {
                    "trend_segment_id": "seg_1",
                    "symbol": SYMBOL,
                    "timeframe": "4h",
                    "start_idx": 70,
                    "end_idx": 90,
                }
            ]
        )

        comparison = tom.build_pre_trend_feature_comparison(
            {(SYMBOL, "4h"): frame},
            segments,
            {tom.symbol_to_inst_id(SYMBOL): pd.DataFrame(columns=["funding_time_utc", "funding_rate"])},
        )

        self.assertFalse(comparison.empty)
        self.assertIn("pre_trend_atr_percentile", set(comparison["feature"]))
        self.assertIn("effect_size", comparison.columns)

    def test_summary_decision_fields_generated(self) -> None:
        history_range = tom.resolve_history_range("2025-01-01", "2025-01-10", "Asia/Shanghai")
        segments = pd.DataFrame(
            [
                {
                    "trend_segment_id": f"seg_{index}",
                    "symbol": SYMBOL,
                    "timeframe": "4h",
                    "direction": "up" if index % 2 == 0 else "down",
                    "start_time": (pd.Timestamp("2025-01-01T03:59:00+08:00") + pd.Timedelta(hours=4 * index)).isoformat(),
                    "end_time": (pd.Timestamp("2025-01-01T07:59:00+08:00") + pd.Timedelta(hours=4 * index)).isoformat(),
                    "duration_bars": 2,
                    "trend_return": 0.02,
                    "mfe": 0.03,
                    "mae": -0.01,
                    "month": "2025-01",
                    "quarter": "2025Q1",
                    "abs_trend_return": 0.02,
                }
                for index in range(3)
            ]
        )
        by_symbol = tom.aggregate_segments(segments, ["symbol"], history_range)
        by_timeframe = tom.aggregate_segments(segments, ["timeframe"], history_range)
        by_month = tom.aggregate_segments(segments, ["month"], history_range)
        by_quarter = tom.aggregate_segments(segments, ["quarter"], history_range)
        summary = tom.build_summary(
            symbols=[SYMBOL],
            timeframes=["4h"],
            history_range=history_range,
            segments=segments,
            by_symbol=by_symbol,
            by_timeframe=by_timeframe,
            by_month=by_month,
            by_quarter=by_quarter,
            legacy_summary=tom.legacy_coverage_summary(pd.DataFrame(columns=tom.LEGACY_COVERAGE_COLUMNS)),
            pre_feature_comparison=pd.DataFrame(columns=["feature", "abs_effect_size"]),
            data_quality={"all_symbols_complete": True},
            funding_quality={"funding_data_complete": False, "missing_inst_ids": []},
            legacy_warnings=[],
        )

        for key in [
            "enough_trend_opportunities",
            "trend_opportunities_are_diversified",
            "legacy_strategies_failed_to_capture_trends",
            "legacy_strategies_trade_too_much_in_nontrend",
            "pre_trend_features_exist",
            "recommended_next_research_direction",
        ]:
            self.assertIn(key, summary)

    def test_output_csv_json_md_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "trend_opportunity_map"
            funding_dir = Path(tmpdir) / "funding"
            funding_dir.mkdir()
            outputs = tom.run_research(
                symbols=[SYMBOL],
                start="2025-01-01",
                end="2025-01-05",
                timezone_name="Asia/Shanghai",
                timeframes=["4h"],
                output_dir=output_dir,
                funding_dir=funding_dir,
                database_path=Path(tmpdir) / "missing.db",
                data_check_strict=True,
                logger=logging.getLogger("test_research_trend_opportunity_map"),
                bars_by_symbol={SYMBOL: make_1m_bars(SYMBOL, 5)},
                legacy_trade_files=[],
            )

            self.assertTrue(outputs.data_quality["all_symbols_complete"])
            for filename in tom.REQUIRED_OUTPUT_FILES:
                self.assertTrue((output_dir / filename).exists(), filename)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-trend-opportunity-map:", makefile)


if __name__ == "__main__":
    unittest.main()
