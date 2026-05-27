from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import research_early_trend_classifier_v1 as etc


TZ = "Asia/Shanghai"
SYMBOL_A = "BTCUSDT_SWAP_OKX.GLOBAL"
SYMBOL_B = "ETHUSDT_SWAP_OKX.GLOBAL"


def make_bars(n: int = 140, trend: float = 1.0, start: str = "2025-01-01T00:00:00+08:00") -> pd.DataFrame:
    open_times = pd.date_range(start, periods=n, freq="4h")
    rows = []
    for index, open_time in enumerate(open_times):
        close = 100.0 + trend * index
        rows.append(
            {
                "open_time": open_time,
                "datetime": open_time + pd.Timedelta(minutes=239),
                "open": close - 0.25,
                "high": close + 1.0 + (index % 3) * 0.1,
                "low": close - 1.0 - (index % 2) * 0.1,
                "close": close,
                "volume": 100.0 + index,
            }
        )
    return pd.DataFrame(rows)


def make_segment(frame: pd.DataFrame, start: int, end: int, direction: str = "up", segment_id: str = "seg") -> dict[str, object]:
    return {
        "trend_segment_id": segment_id,
        "symbol": SYMBOL_A,
        "timeframe": "4h",
        "direction": direction,
        "start_time": frame.iloc[start]["datetime"].isoformat(),
        "end_time": frame.iloc[end]["datetime"].isoformat(),
        "start_idx": start,
        "end_idx": end,
        "duration_bars": end - start + 1,
        "trend_return": 0.10,
        "atr_labels": f"{direction}trend_2atr",
        "efficiency_labels": "trend_efficiency_20_ge_0.60",
    }


def make_scored_dataset(n: int = 80) -> pd.DataFrame:
    bars = make_bars(n)
    rows = []
    for index, row in bars.iterrows():
        split = "train_ext" if index < 40 else "validation_ext" if index < 60 else "oos_ext"
        rows.append(
            {
                "timestamp": row["datetime"],
                "symbol": SYMBOL_A,
                "inst_id": etc.symbol_to_inst_id(SYMBOL_A),
                "timeframe": "4h",
                "bar_index": index,
                "split": split,
                "label": "early_uptrend" if index % 10 == 0 else "nontrend",
                "direction_proxy": "long",
                "direction_match": index % 10 == 0,
                "early_trend_score": float(index),
                "early_trend_score_component_count": 4,
                "close": row["close"],
                "atr_20": 1.0,
                "forward_return_4h": 0.01,
                "forward_return_1d": 0.02,
                "forward_return_3d": 0.03,
                "future_2atr_hit": True,
                "future_3atr_hit": index % 2 == 0,
            }
        )
    return pd.DataFrame(rows)


def make_gate_event_summary(pass_case: bool = True) -> pd.DataFrame:
    rows = []
    for split in etc.SPLIT_NAMES:
        rate = 0.30 if split == "train_ext" else 0.25
        no_cost = 10.0 if pass_case else -1.0
        rows.append(
            {
                "group": "A",
                "event_group": etc.EVENT_GROUPS["A"],
                "hold": "hold_4h",
                "split": split,
                "event_count": 30,
                "trade_count": 30,
                "early_trend_rate": rate,
                "early_uptrend_rate": rate,
                "early_downtrend_rate": 0.0,
                "nontrend_rate": 1.0 - rate,
                "direction_match_rate": 1.0,
                "future_2atr_hit_rate": 0.5,
                "future_3atr_hit_rate": 0.4,
                "avg_forward_return_3d": 0.01,
                "no_cost_pnl": no_cost,
                "cost_aware_pnl": 1.0 if pass_case else -1.0,
                "funding_pnl": 0.0,
                "funding_adjusted_pnl": 1.0 if pass_case else -1.0,
            }
        )
    return pd.DataFrame(rows)


class ResearchEarlyTrendClassifierV1Test(unittest.TestCase):
    def test_early_trend_label_generation(self) -> None:
        frame = make_bars(40)
        segments = etc.normalize_trend_segments(pd.DataFrame([make_segment(frame, 4, 15, "up")]), TZ)
        labels = etc.build_early_trend_labels_for_frame(frame, SYMBOL_A, "4h", segments, future_window_bars=2, boundary_buffer_bars=0)

        self.assertEqual(labels.iloc[4]["label"], "early_uptrend")
        self.assertEqual(labels.iloc[6]["label"], "early_uptrend")

    def test_middle_late_and_nontrend_label_generation(self) -> None:
        frame = make_bars(40)
        segments = etc.normalize_trend_segments(pd.DataFrame([make_segment(frame, 4, 15, "up")]), TZ)
        labels = etc.build_early_trend_labels_for_frame(frame, SYMBOL_A, "4h", segments, future_window_bars=2, boundary_buffer_bars=0)

        self.assertEqual(labels.iloc[10]["label"], "middle_trend")
        self.assertEqual(labels.iloc[14]["label"], "late_trend")
        self.assertEqual(labels.iloc[20]["label"], "nontrend")

    def test_ambiguous_overlap_exclusion(self) -> None:
        frame = make_bars(40)
        segments = etc.normalize_trend_segments(
            pd.DataFrame([make_segment(frame, 4, 15, "up", "a"), make_segment(frame, 8, 18, "down", "b")]),
            TZ,
        )
        labels = etc.build_early_trend_labels_for_frame(frame, SYMBOL_A, "4h", segments, future_window_bars=2, boundary_buffer_bars=0)

        self.assertEqual(labels.iloc[9]["label"], "excluded_ambiguous")

    def test_feature_does_not_use_future_data(self) -> None:
        frame = make_bars(140)
        changed = frame.copy()
        changed.loc[139, "close"] = 10000.0

        original = etc.add_symbol_features(frame, SYMBOL_A, "4h")
        modified = etc.add_symbol_features(changed, SYMBOL_A, "4h")

        self.assertAlmostEqual(float(original.loc[80, "trend_efficiency_20"]), float(modified.loc[80, "trend_efficiency_20"]))

    def test_trend_efficiency_feature_calculation(self) -> None:
        features = etc.add_symbol_features(make_bars(80, trend=1.0), SYMBOL_A, "4h")

        self.assertAlmostEqual(float(features.loc[25, "trend_efficiency_20"]), 1.0)

    def test_breadth_feature_calculation(self) -> None:
        a = etc.add_symbol_features(make_bars(80, trend=1.0), SYMBOL_A, "4h")
        b = etc.add_symbol_features(make_bars(80, trend=-1.0), SYMBOL_B, "4h")
        dataset = etc.add_cross_symbol_features(pd.concat([a, b], ignore_index=True), [SYMBOL_A, SYMBOL_B], ["4h"])
        last_time = dataset["timestamp"].max()
        row = dataset[(dataset["symbol"] == SYMBOL_A) & (dataset["timestamp"] == last_time)].iloc[0]

        self.assertEqual(int(row["positive_return_symbol_count_4h"]), 1)

    def test_rank_feature_calculation(self) -> None:
        a = etc.add_symbol_features(make_bars(80, trend=2.0), SYMBOL_A, "4h")
        b = etc.add_symbol_features(make_bars(80, trend=0.2), SYMBOL_B, "4h")
        dataset = etc.add_cross_symbol_features(pd.concat([a, b], ignore_index=True), [SYMBOL_A, SYMBOL_B], ["4h"])
        last_time = dataset["timestamp"].max()
        row = dataset[(dataset["symbol"] == SYMBOL_A) & (dataset["timestamp"] == last_time)].iloc[0]

        self.assertAlmostEqual(float(row["return_rank_20"]), 1.0)

    def test_volatility_feature_calculation(self) -> None:
        features = etc.add_symbol_features(make_bars(140, trend=1.0), SYMBOL_A, "4h")

        self.assertIn("atr_percentile_100", features.columns)
        self.assertTrue(pd.to_numeric(features["range_width_percentile_100"], errors="coerce").notna().any())

    def test_funding_feature_alignment(self) -> None:
        frame = etc.add_symbol_features(make_bars(10), SYMBOL_A, "4h")
        funding = pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(["2024-12-31T16:00:00Z", "2025-01-01T08:00:00Z"]),
                "funding_rate": [0.001, 0.002],
            }
        )
        aligned = etc.align_funding_features(frame, {etc.symbol_to_inst_id(SYMBOL_A): funding}, [SYMBOL_A], ["4h"], True)

        self.assertAlmostEqual(float(aligned.iloc[-1]["funding_rate"]), 0.002)

    def test_feature_bucket_uses_train_boundaries(self) -> None:
        dataset = self._make_bucket_dataset()
        buckets, _predictiveness = etc.build_feature_bucket_analysis(dataset, ["trend_efficiency_20"])
        train_high = buckets[buckets["feature"] == "trend_efficiency_20"]["train_bucket_boundary_high"].max()

        self.assertLessEqual(float(train_high), 9.0)

    def test_validation_and_oos_do_not_participate_in_bucket_boundaries(self) -> None:
        dataset = self._make_bucket_dataset()
        buckets, _predictiveness = etc.build_feature_bucket_analysis(dataset, ["trend_efficiency_20"])
        validation_top = buckets[
            (buckets["split"] == "validation_ext")
            & (buckets["feature"] == "trend_efficiency_20")
            & (buckets["bucket"] == "q80-q100")
        ].iloc[0]

        self.assertLessEqual(float(validation_top["train_bucket_boundary_high"]), 9.0)

    def test_composite_score_calculation(self) -> None:
        dataset = self._make_composite_dataset()
        scored = etc.compute_composite_score(dataset, [])

        self.assertTrue(scored["composite_score_available"].any())
        self.assertTrue(pd.to_numeric(scored["early_trend_score"], errors="coerce").notna().any())

    def test_top_score_event_generation(self) -> None:
        scored = make_scored_dataset()
        events = etc.generate_top_score_events(scored, {(SYMBOL_A, "4h"): make_bars(100)}, etc.EtcConfig())

        self.assertIn("A", set(events["group"]))
        self.assertIn("B", set(events["group"]))
        self.assertIn("C", set(events["group"]))

    def test_random_control_generation(self) -> None:
        scored = make_scored_dataset()
        events = etc.generate_top_score_events(scored, {(SYMBOL_A, "4h"): make_bars(100)}, etc.EtcConfig())

        self.assertIn("D", set(events["group"]))

    def test_reverse_test_generation(self) -> None:
        scored = make_scored_dataset()
        events = etc.generate_top_score_events(scored, {(SYMBOL_A, "4h"): make_bars(100)}, etc.EtcConfig())

        self.assertIn("E", set(events["group"]))
        reverse = events[events["group"] == "E"].iloc[0]
        self.assertEqual(reverse["direction"], "short")

    def test_cost_aware_calculation(self) -> None:
        scored = make_scored_dataset()
        events = etc.generate_top_score_events(scored, {(SYMBOL_A, "4h"): make_bars(100, trend=2.0)}, etc.EtcConfig()).head(1)
        trades = etc.simulate_event_trades(events, {(SYMBOL_A, "4h"): make_bars(100, trend=2.0)}, {}, etc.EtcConfig())
        trade = trades[trades["hold"] == "hold_4h"].iloc[0]

        self.assertAlmostEqual(float(trade["cost_aware_pnl"]), float(trade["no_cost_pnl"]) - 2.0)

    def test_funding_adjusted_calculation(self) -> None:
        scored = make_scored_dataset()
        bars = make_bars(100, trend=2.0)
        events = etc.generate_top_score_events(scored, {(SYMBOL_A, "4h"): bars}, etc.EtcConfig()).head(1)
        funding = pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(["2025-01-02T00:00:00Z"]),
                "funding_rate": [-0.001],
            }
        )
        trades = etc.simulate_event_trades(events, {(SYMBOL_A, "4h"): bars}, {etc.symbol_to_inst_id(SYMBOL_A): funding}, etc.EtcConfig())
        trade = trades[trades["hold"] == "hold_1d"].iloc[0]

        self.assertGreaterEqual(float(trade["funding_adjusted_pnl"]), float(trade["cost_aware_pnl"]))

    def test_concentration_calculation(self) -> None:
        trades = pd.DataFrame(
            [
                {"group": "A", "event_group": "top10", "hold": "hold_4h", "split": "oos_ext", "symbol": SYMBOL_A, "funding_adjusted_pnl": 5.0},
                {"group": "A", "event_group": "top10", "hold": "hold_4h", "split": "oos_ext", "symbol": SYMBOL_B, "funding_adjusted_pnl": 5.0},
            ]
        )
        concentration = etc.build_concentration_summary(trades)

        self.assertAlmostEqual(float(concentration.iloc[0]["largest_symbol_pnl_share"]), 0.5)

    def test_phase_gate_true_case(self) -> None:
        event_summary = make_gate_event_summary(True)
        random_control = pd.DataFrame(
            [
                {
                    "group": "A",
                    "event_group": etc.EVENT_GROUPS["A"],
                    "hold": "hold_4h",
                    "split": split,
                    "random_early_trend_rate": 0.10,
                    "random_weaker": True,
                }
                for split in etc.SPLIT_NAMES
            ]
        )
        reverse_test = pd.DataFrame([{"hold": "hold_4h", "split": "oos_ext", "reverse_weaker": True}])
        concentration = pd.DataFrame(
            [{"group": "A", "hold": "hold_4h", "split": split, "concentration_pass": True} for split in etc.SPLIT_NAMES]
        )
        gates, _rejected = etc.evaluate_phase_gates(event_summary, random_control, reverse_test, concentration, True)

        self.assertTrue(gates["can_enter_phase2"])

    def test_phase_gate_false_case(self) -> None:
        event_summary = make_gate_event_summary(False)
        random_control = pd.DataFrame(
            [{"group": "A", "hold": "hold_4h", "split": split, "random_early_trend_rate": 0.30, "random_weaker": False} for split in etc.SPLIT_NAMES]
        )
        reverse_test = pd.DataFrame([{"hold": "hold_4h", "split": "oos_ext", "reverse_weaker": False}])
        concentration = pd.DataFrame(
            [{"group": "A", "hold": "hold_4h", "split": split, "concentration_pass": False} for split in etc.SPLIT_NAMES]
        )
        gates, _rejected = etc.evaluate_phase_gates(event_summary, random_control, reverse_test, concentration, True)

        self.assertFalse(gates["can_enter_phase2"])

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = {
                "can_enter_phase2": False,
                "strategy_development_allowed": False,
                "demo_live_allowed": False,
                "final_decision": "postmortem_or_pause",
            }
            etc.write_outputs(
                output_dir,
                data_quality={"market_data_complete": True},
                labels=pd.DataFrame(columns=etc.LABEL_COLUMNS),
                feature_dataset=pd.DataFrame(),
                feature_bucket_analysis=pd.DataFrame(),
                feature_predictiveness=pd.DataFrame(),
                scored=pd.DataFrame(),
                score_bucket_analysis=pd.DataFrame(),
                events=pd.DataFrame(),
                trades=pd.DataFrame(),
                event_summary=pd.DataFrame(),
                by_symbol=pd.DataFrame(),
                by_timeframe=pd.DataFrame(),
                by_split=pd.DataFrame(),
                concentration=pd.DataFrame(),
                reverse_test=pd.DataFrame(),
                random_control=pd.DataFrame(),
                funding_summary=pd.DataFrame(),
                rejected_reasons=pd.DataFrame(),
                summary=summary,
            )

            self.assertTrue((output_dir / "etc_v1_summary.json").exists())
            self.assertTrue((output_dir / "etc_v1_report.md").exists())
            self.assertTrue((output_dir / "early_trend_labels.csv").exists())

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-etc-v1:", makefile)

    def _make_bucket_dataset(self) -> pd.DataFrame:
        rows = []
        for split, offset in [("train_ext", 0), ("validation_ext", 100), ("oos_ext", 200)]:
            for index in range(10):
                rows.append(
                    {
                        "split": split,
                        "label": "early_uptrend" if index >= 8 else "nontrend",
                        "trend_efficiency_20": float(offset + index),
                        "direction_match": True,
                        "forward_return_4h": 0.0,
                        "forward_return_1d": 0.0,
                        "forward_return_3d": 0.0,
                        "future_2atr_hit": False,
                        "future_3atr_hit": False,
                    }
                )
        return pd.DataFrame(rows)

    def _make_composite_dataset(self) -> pd.DataFrame:
        rows = []
        for split in etc.SPLIT_NAMES:
            for index in range(12):
                rows.append(
                    {
                        "split": split,
                        "trend_efficiency_change_20": float(index),
                        "breadth_acceleration": float(index) / 10.0,
                        "rank_change_20": float(index) / 20.0,
                        "efficiency_to_volatility_ratio_20": float(index) + 1.0,
                        "funding_overheat_score": 0.1,
                        "volatility_noise_score": 0.2,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
