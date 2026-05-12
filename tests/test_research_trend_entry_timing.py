from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import research_trend_entry_timing as tet
import research_trend_capture_exit_convexity as tce


SYMBOL = "AAAUSDT_SWAP_OKX.GLOBAL"


def make_segment() -> pd.Series:
    segments = tce.normalize_segments(
        pd.DataFrame(
            [
                {
                    "trend_segment_id": "seg_1",
                    "symbol": SYMBOL,
                    "timeframe": "4h",
                    "direction": "up",
                    "start_time": "2025-01-02T00:00:00+08:00",
                    "end_time": "2025-01-03T00:00:00+08:00",
                    "start_price": 100.0,
                    "end_price": 120.0,
                    "duration_bars": 6,
                    "abs_trend_return": 0.2,
                }
            ]
        ),
        "Asia/Shanghai",
    )
    return segments.iloc[0]


def make_closed_frame(periods: int = 90, start: str = "2025-01-01T00:00:00+08:00", step: float = 0.5) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp(start)
    for index in range(periods):
        close = 100.0 + index * step
        rows.append(
            {
                "datetime": base + pd.Timedelta(hours=4 * index),
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1.0,
                "atr14": 2.0,
            }
        )
    frame = tet.add_basic_indicators(pd.DataFrame(rows))
    frame = tet.attach_funding_to_frame(frame, pd.DataFrame(columns=["funding_time_utc", "funding_rate"]))
    frame["symbol"] = SYMBOL
    frame["timeframe"] = "4h"
    return frame


def add_market_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["positive_ret3_count"] = 3
    result["negative_ret3_count"] = 3
    result["above_median_count"] = 3
    result["below_median_count"] = 3
    result["market_ret3_mean"] = 0.01
    result["dispersion_ret3"] = 0.01
    result["funding_dispersion"] = 0.0
    result["positive_funding_count"] = 2
    result["negative_funding_count"] = 2
    result["avg_pairwise_corr_20"] = 0.5
    result["avg_pairwise_corr_20_rising"] = True
    result["rs_rank_long_20"] = 1
    result["rs_rank_long_55"] = 1
    result["rs_rank_short_20"] = 1
    result["rs_rank_short_55"] = 1
    result["range_high_55_prev"] = result["close"] + 10.0
    result["range_low_55_prev"] = result["close"] - 10.0
    result["mid_20_prev"] = result["close"] - 1.0
    return result


def make_minute_bars(days: int = 120) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-01-01T00:00:00+08:00")
    for index in range(days * 1440):
        close = 100.0 + index * 0.001
        rows.append(
            {
                "datetime": start + pd.Timedelta(minutes=index),
                "open": close - 0.001,
                "high": close + 0.002,
                "low": close - 0.002,
                "close": close,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows)


class TrendEntryTimingResearchTest(unittest.TestCase):
    def test_entry_phase_classification(self) -> None:
        self.assertEqual(tet.classify_entry_phase(-0.1, True), "pre_trend")
        self.assertEqual(tet.classify_entry_phase(0.05, True), "first_10pct")
        self.assertEqual(tet.classify_entry_phase(0.20, True), "first_25pct")
        self.assertEqual(tet.classify_entry_phase(0.50, True), "middle_25_75pct")
        self.assertEqual(tet.classify_entry_phase(0.90, True), "late_75pct_plus")
        self.assertEqual(tet.classify_entry_phase(1.10, True), "after_trend")
        self.assertEqual(tet.classify_entry_phase(None, False), "nontrend")

    def test_missed_mfe_before_entry_calculation(self) -> None:
        frame = make_closed_frame(periods=20, start="2025-01-02T00:00:00+08:00", step=2.0)
        segment = make_segment()

        missed, remaining = tet.segment_path_metrics(
            frame,
            segment,
            "long",
            pd.Timestamp("2025-01-02T12:00:00+08:00"),
            106.0,
        )

        self.assertIsNotNone(missed)
        self.assertIsNotNone(remaining)
        self.assertGreater(float(missed), 0.0)
        self.assertGreater(float(remaining), 0.0)

    def test_legacy_entry_timing_diagnostics(self) -> None:
        frame = make_closed_frame(periods=40)
        segments = pd.DataFrame([make_segment()])
        segments_by_key = tet.build_segments_by_key(segments, ["4h"])
        trade = pd.Series(
            {
                "strategy_source": "trend_v3_extended",
                "policy_or_group": "policy",
                "symbol": SYMBOL,
                "trade_timeframe": "4h",
                "trade_id": "t1",
                "split": "train_ext",
                "direction": "long",
                "entry_time": "2025-01-02T16:00:00+08:00",
                "entry_ts": pd.Timestamp("2025-01-02T16:00:00+08:00"),
                "entry_price": 108.0,
                "exit_time": "2025-01-02T20:00:00+08:00",
                "exit_ts": pd.Timestamp("2025-01-02T20:00:00+08:00"),
            }
        )

        diagnostics = tet.build_legacy_entry_diagnostics(pd.DataFrame([trade]), segments_by_key, {(SYMBOL, "4h"): frame})

        self.assertEqual(len(diagnostics.index), 1)
        self.assertIn("entry_lag_pct_of_segment", diagnostics.columns)
        self.assertTrue(bool(diagnostics.iloc[0]["direction_matches_segment"]))

    def test_candidate_event_generation(self) -> None:
        frame = add_market_columns(make_closed_frame())
        frames = {(SYMBOL, "4h"): frame}
        market = {"4h": frame[["datetime", "positive_ret3_count", "negative_ret3_count", "above_median_count", "below_median_count", "market_ret3_mean", "dispersion_ret3", "funding_dispersion", "positive_funding_count", "negative_funding_count", "avg_pairwise_corr_20", "avg_pairwise_corr_20_rising"]]}
        thresholds = {
            "4h": {
                "ret3_long": -1.0,
                "ret3_short": 1.0,
                "ret6_long": -1.0,
                "ret6_short": 1.0,
                "eff_delta": -1.0,
                "vol_max": 1.0,
                "funding_abs_max": 1.0,
                "breadth_long_min": 2.0,
                "breadth_short_min": 2.0,
                "dispersion_max": 1.0,
                "funding_dispersion_max": 1.0,
                "funding_sign_crowd_max": 5.0,
            }
        }

        events = tet.generate_candidate_events(frames, market, thresholds, [SYMBOL], ["4h"], "Asia/Shanghai")

        self.assertFalse(events.empty)
        self.assertIn("pre_breakout_momentum_acceleration", set(events["family"]))

    def test_breakout_retest_reclaim_event(self) -> None:
        frame = make_closed_frame(periods=30)
        frame["range_high_20_prev"] = 105.0
        frame["range_low_20_prev"] = 95.0
        frame.loc[20, "close"] = 107.0
        frame.loc[21, "low"] = 104.8
        frame.loc[21, "close"] = 106.0

        rows = tet.generate_breakout_retest_events(frame, "breakout_retest_reclaim", SYMBOL, "4h", "Asia/Shanghai")

        self.assertTrue(rows)
        self.assertEqual(rows[0]["family"], "breakout_retest_reclaim")

    def test_cross_symbol_breadth_acceleration_event(self) -> None:
        frame = add_market_columns(make_closed_frame())
        frame["positive_ret3_count"] = [1 if index < 30 else 3 for index in range(len(frame.index))]
        frame["negative_ret3_count"] = [1 if index < 30 else 3 for index in range(len(frame.index))]
        frames = {(SYMBOL, "4h"): frame}
        market = {"4h": frame[[column for column in frame.columns if column in {"datetime", "positive_ret3_count", "negative_ret3_count", "above_median_count", "below_median_count", "market_ret3_mean", "dispersion_ret3", "funding_dispersion", "positive_funding_count", "negative_funding_count", "avg_pairwise_corr_20", "avg_pairwise_corr_20_rising"}]]}
        thresholds = tet.train_thresholds(frames, market, "Asia/Shanghai")
        thresholds["4h"]["breadth_long_min"] = 2.0
        thresholds["4h"]["dispersion_max"] = 1.0
        events = tet.generate_candidate_events(frames, market, thresholds, [SYMBOL], ["4h"], "Asia/Shanghai")

        self.assertIn("cross_symbol_breadth_acceleration", set(events["family"]))

    def test_funding_neutral_momentum_event(self) -> None:
        frame = add_market_columns(make_closed_frame())
        frames = {(SYMBOL, "4h"): frame}
        market = {"4h": frame[["datetime", "positive_ret3_count", "negative_ret3_count", "above_median_count", "below_median_count", "market_ret3_mean", "dispersion_ret3", "funding_dispersion", "positive_funding_count", "negative_funding_count", "avg_pairwise_corr_20", "avg_pairwise_corr_20_rising"]]}
        thresholds = tet.train_thresholds(frames, market, "Asia/Shanghai")
        thresholds["4h"]["ret3_long"] = -1.0
        thresholds["4h"]["funding_abs_max"] = 1.0
        events = tet.generate_candidate_events(frames, market, thresholds, [SYMBOL], ["4h"], "Asia/Shanghai")

        self.assertIn("funding_neutral_momentum", set(events["family"]))

    def test_relative_strength_leader_event(self) -> None:
        frame = add_market_columns(make_closed_frame())
        frames = {(SYMBOL, "4h"): frame}
        market = {"4h": frame[["datetime", "positive_ret3_count", "negative_ret3_count", "above_median_count", "below_median_count", "market_ret3_mean", "dispersion_ret3", "funding_dispersion", "positive_funding_count", "negative_funding_count", "avg_pairwise_corr_20", "avg_pairwise_corr_20_rising"]]}
        thresholds = tet.train_thresholds(frames, market, "Asia/Shanghai")
        thresholds["4h"]["funding_abs_max"] = 1.0
        events = tet.generate_candidate_events(frames, market, thresholds, [SYMBOL], ["4h"], "Asia/Shanghai")

        self.assertIn("relative_strength_leader", set(events["family"]))

    def test_split_thresholds_use_train_only(self) -> None:
        train_frame = add_market_columns(make_closed_frame(periods=120, start="2023-01-01T00:00:00+08:00", step=0.1))
        validation_frame = add_market_columns(make_closed_frame(periods=120, start="2024-07-01T00:00:00+08:00", step=10.0))
        frame = pd.concat([train_frame, validation_frame], ignore_index=True)
        frames = {(SYMBOL, "4h"): frame}
        market = {"4h": frame[["datetime", "positive_ret3_count", "negative_ret3_count", "above_median_count", "below_median_count", "market_ret3_mean", "dispersion_ret3", "funding_dispersion", "positive_funding_count", "negative_funding_count", "avg_pairwise_corr_20", "avg_pairwise_corr_20_rising"]]}

        thresholds = tet.train_thresholds(frames, market, "Asia/Shanghai")

        self.assertLess(thresholds["4h"]["ret3_long"], 0.1)

    def test_reverse_and_random_controls(self) -> None:
        frame = add_market_columns(make_closed_frame(periods=120, start="2023-01-01T00:00:00+08:00"))
        event = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "family": "relative_strength_leader",
                    "symbol": SYMBOL,
                    "timeframe": "4h",
                    "direction": "long",
                    "event_time": frame.iloc[60]["datetime"].isoformat(),
                    "event_price": float(frame.iloc[60]["close"]),
                    "split": "train_ext",
                    "trend_segment_id": "seg_1",
                    "direction_matches_segment": True,
                    "entry_phase": "first_10pct",
                    "entry_lag_bars": 0.0,
                    "entry_lag_pct_of_segment": 0.0,
                    "missed_mfe_before_entry": 0.0,
                    "remaining_mfe_after_entry": 0.1,
                }
            ]
        )
        funding = {tce.symbol_to_inst_id(SYMBOL): tce.FundingIndex(np.array([], dtype="int64"), np.array([0.0]), pd.Timestamp("2023-01-01T00:00:00Z"), pd.Timestamp("2023-01-01T00:00:00Z"))}

        reverse = tet.build_reverse_tests(event, {(SYMBOL, "4h"): frame}, funding, {"relative_strength_leader": "fixed_hold_4h"})
        random_control = tet.build_random_controls(event, {(SYMBOL, "4h"): frame}, funding, {"relative_strength_leader": "fixed_hold_4h"}, "Asia/Shanghai")

        self.assertFalse(reverse.empty)
        self.assertFalse(random_control.empty)

    def test_stable_gate_true_case(self) -> None:
        row = {
            "family": "candidate",
            "selected_hold_label": "fixed_hold_4h",
            "trend_segment_recall": 0.3,
            "early_entry_rate": 0.5,
            "direction_match_rate": 0.6,
            "funding_adjusted_pnl": 100.0,
            "reverse_test_result": -10.0,
            "random_time_control_result": -5.0,
            "largest_symbol_pnl_share": 0.5,
            "top_5pct_trade_pnl_contribution": 0.5,
        }
        for split in tet.SPLITS:
            row[f"{split}_trade_count"] = 10
            row[f"{split}_no_cost_pnl"] = 1.0
            row[f"{split}_cost_aware_pnl"] = 1.0
            row[f"{split}_funding_adjusted_pnl"] = 1.0

        rejected, candidates = tet.evaluate_stable_gates(pd.DataFrame([row]))

        self.assertTrue(candidates)
        self.assertTrue(bool(rejected.iloc[0]["stable_like"]))

    def test_stable_gate_false_case(self) -> None:
        row = {
            "family": "candidate",
            "selected_hold_label": "fixed_hold_4h",
            "trend_segment_recall": 0.1,
            "early_entry_rate": 0.2,
            "direction_match_rate": 0.4,
            "funding_adjusted_pnl": -1.0,
            "reverse_test_result": 1.0,
            "random_time_control_result": 1.0,
            "largest_symbol_pnl_share": 0.9,
            "top_5pct_trade_pnl_contribution": 0.9,
        }
        for split in tet.SPLITS:
            row[f"{split}_trade_count"] = 1
            row[f"{split}_no_cost_pnl"] = -1.0
            row[f"{split}_cost_aware_pnl"] = -1.0
            row[f"{split}_funding_adjusted_pnl"] = -1.0

        rejected, candidates = tet.evaluate_stable_gates(pd.DataFrame([row]))

        self.assertFalse(candidates)
        self.assertFalse(bool(rejected.iloc[0]["stable_like"]))

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trend_map = root / "trend_map"
            trend_map.mkdir()
            segment = make_segment().to_frame().T.drop(columns=["start_ts", "end_ts"])
            segment.to_csv(trend_map / "trend_segments.csv", index=False)
            (trend_map / "trend_opportunity_summary.json").write_text(
                json.dumps(
                    {
                        "enough_trend_opportunities": True,
                        "trend_opportunities_are_diversified": True,
                        "data_quality": {"all_symbols_complete": True, "funding_data_complete": True},
                        "legacy_analysis": {"main_failure_mode": "entered_middle_or_late"},
                    }
                ),
                encoding="utf-8",
            )
            (trend_map / "data_quality.json").write_text(json.dumps({"all_symbols_complete": True}), encoding="utf-8")

            trend_v3 = root / "trend_v3"
            for split in tet.SPLITS:
                split_dir = trend_v3 / split
                split_dir.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "policy_name": "policy",
                            "symbol": SYMBOL,
                            "direction": "long",
                            "entry_time": "2025-01-02T12:00:00+08:00",
                            "entry_price": 103.0,
                            "exit_time": "2025-01-02T20:00:00+08:00",
                            "exit_price": 105.0,
                            "holding_minutes": 480,
                            "volume": 1.0,
                            "contract_size": 1.0,
                            "no_cost_pnl": 2.0,
                            "net_pnl": 1.9,
                            "fee": 0.1,
                            "slippage": 0.0,
                            "timeframe": "4h",
                        }
                    ]
                ).to_csv(split_dir / "trend_v3_trades.csv", index=False)
            funding = root / "funding"
            funding.mkdir()
            pd.DataFrame(
                {
                    "inst_id": [tce.symbol_to_inst_id(SYMBOL)],
                    "funding_time_utc": ["2025-01-01T00:00:00Z"],
                    "funding_rate": [0.0],
                }
            ).to_csv(funding / f"{tce.symbol_to_inst_id(SYMBOL)}_funding_test.csv", index=False)

            outputs = tet.run_research(
                trend_map_dir=trend_map,
                trend_v3_dir=trend_v3,
                funding_dir=funding,
                output_dir=root / "out",
                symbols=[SYMBOL],
                start="2025-01-01",
                end="2025-04-30",
                timezone_name="Asia/Shanghai",
                timeframes=["4h"],
                data_check_strict=True,
                logger=logging.getLogger("test_tet"),
                database_path=root / "missing.db",
                bars_by_symbol={SYMBOL: make_minute_bars(days=130)},
            )

            self.assertTrue(outputs.output_dir.exists())
            for filename in tet.REQUIRED_OUTPUT_FILES:
                self.assertTrue((outputs.output_dir / filename).exists(), filename)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-trend-entry-timing:", makefile)
        self.assertIn("scripts/research_trend_entry_timing.py", makefile)


if __name__ == "__main__":
    unittest.main()
