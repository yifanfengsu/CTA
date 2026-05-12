from __future__ import annotations

import json
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

import research_trend_capture_exit_convexity as tce


SYMBOL = "AAAUSDT_SWAP_OKX.GLOBAL"


def make_internal_trade(
    *,
    entry: str = "2025-01-01T04:00:00+08:00",
    exit_time: str = "2025-01-01T12:00:00+08:00",
    direction: str = "long",
    split: str = "train_ext",
    no_cost: float = 1.0,
) -> pd.Series:
    entry_ts = pd.Timestamp(entry)
    exit_ts = pd.Timestamp(exit_time)
    return pd.Series(
        {
            "strategy_source": "unit",
            "policy_or_group": "policy",
            "symbol": SYMBOL,
            "trade_timeframe": "4h",
            "trade_id": "t1",
            "split": split,
            "direction": direction,
            "entry_time": entry_ts.isoformat(),
            "entry_ts": entry_ts,
            "entry_price": 100.0,
            "exit_time": exit_ts.isoformat(),
            "exit_ts": exit_ts,
            "exit_price": 102.0 if direction == "long" else 98.0,
            "holding_minutes": (exit_ts - entry_ts).total_seconds() / 60.0,
            "no_cost_pnl": no_cost,
            "cost_aware_pnl": no_cost - 0.1,
            "cost_drag": 0.1,
            "pnl_multiplier": 1.0,
            "notional": 100.0,
        }
    )


def make_segments() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "trend_segment_id": "seg_1",
                "symbol": SYMBOL,
                "timeframe": "4h",
                "direction": "up",
                "start_time": "2025-01-01T00:00:00+08:00",
                "end_time": "2025-01-02T00:00:00+08:00",
                "duration_bars": 6,
                "abs_trend_return": 0.20,
            }
        ]
    )
    return tce.normalize_segments(raw, "Asia/Shanghai")


def make_closed_bars(closes: list[float], *, direction: str = "long") -> pd.DataFrame:
    times = pd.date_range("2025-01-01T04:00:00+08:00", periods=len(closes), freq="4h")
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "datetime": times[index],
                "open": close,
                "high": close + (1.0 if direction == "long" else 0.5),
                "low": close - (0.5 if direction == "long" else 1.0),
                "close": close,
                "volume": 1.0,
                "atr14": 2.0,
            }
        )
    return pd.DataFrame(rows)


def make_minute_bars(start: str = "2025-01-01T00:00:00+08:00", minutes: int = 4320) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    rows = []
    for index in range(minutes):
        close = 100.0 + index * 0.01
        rows.append(
            {
                "datetime": start_ts + pd.Timedelta(minutes=index),
                "open": close - 0.01,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows)


class ResearchTrendCaptureExitConvexityTest(unittest.TestCase):
    def test_trade_aligns_to_trend_segment(self) -> None:
        segments_by_symbol = tce.build_segments_by_symbol(make_segments(), "4h")
        diag = tce.compute_trade_diagnostic(make_internal_trade(), pd.Timestamp("2025-01-01T12:00:00+08:00"), segments_by_symbol, "4h")

        self.assertTrue(diag["entered_trend_segment"])
        self.assertEqual(diag["trend_segment_id"], "seg_1")

    def test_entry_phase_identification(self) -> None:
        start = pd.Timestamp("2025-01-01T00:00:00+08:00")
        end = pd.Timestamp("2025-01-01T09:00:00+08:00")

        self.assertEqual(tce.classify_entry_phase(pd.Timestamp("2024-12-31T23:00:00+08:00"), start, end), "before_trend")
        self.assertEqual(tce.classify_entry_phase(pd.Timestamp("2025-01-01T01:00:00+08:00"), start, end), "early_trend")
        self.assertEqual(tce.classify_entry_phase(pd.Timestamp("2025-01-01T04:00:00+08:00"), start, end), "middle_trend")
        self.assertEqual(tce.classify_entry_phase(pd.Timestamp("2025-01-01T08:00:00+08:00"), start, end), "late_trend")
        self.assertEqual(tce.classify_entry_phase(pd.Timestamp("2025-01-01T10:00:00+08:00"), start, end), "after_trend")

    def test_exit_phase_identification(self) -> None:
        start = pd.Timestamp("2025-01-01T00:00:00+08:00")
        end = pd.Timestamp("2025-01-02T00:00:00+08:00")

        self.assertEqual(tce.classify_exit_phase(pd.Timestamp("2025-01-01T08:00:00+08:00"), start, end), "before_trend_end")
        self.assertEqual(tce.classify_exit_phase(pd.Timestamp("2025-01-01T22:30:00+08:00"), start, end), "near_trend_end")
        self.assertEqual(tce.classify_exit_phase(pd.Timestamp("2025-01-02T08:00:00+08:00"), start, end), "after_trend_end")

    def test_captured_fraction_calculation(self) -> None:
        fraction = tce.overlap_fraction(
            pd.Timestamp("2025-01-01T06:00:00+08:00"),
            pd.Timestamp("2025-01-01T18:00:00+08:00"),
            pd.Timestamp("2025-01-01T00:00:00+08:00"),
            pd.Timestamp("2025-01-02T00:00:00+08:00"),
        )

        self.assertAlmostEqual(fraction, 0.5)

    def test_early_exit_flag(self) -> None:
        segments_by_symbol = tce.build_segments_by_symbol(make_segments(), "4h")
        diag = tce.compute_trade_diagnostic(
            make_internal_trade(exit_time="2025-01-01T08:00:00+08:00"),
            pd.Timestamp("2025-01-01T08:00:00+08:00"),
            segments_by_symbol,
            "4h",
        )

        self.assertTrue(diag["early_exit_flag"])

    def test_fixed_hold_longer_2x(self) -> None:
        trade = make_internal_trade(entry="2025-01-01T00:00:00+08:00", exit_time="2025-01-01T04:00:00+08:00")
        bars = make_closed_bars([100.0, 101.0, 102.0, 103.0])

        exit_result = tce.simulate_fixed_hold_exit(trade, bars, 2.0, 10)

        self.assertEqual(exit_result.exit_time, pd.Timestamp("2025-01-01T08:00:00+08:00"))

    def test_atr_chandelier_long_and_short(self) -> None:
        long_trade = make_internal_trade(entry="2025-01-01T00:00:00+08:00", direction="long")
        long_bars = make_closed_bars([103.0, 106.0, 108.0, 101.0])
        long_exit = tce.simulate_atr_chandelier_exit(long_trade, long_bars, 3.0)
        self.assertEqual(long_exit.exit_time, pd.Timestamp("2025-01-01T16:00:00+08:00"))

        short_trade = make_internal_trade(entry="2025-01-01T00:00:00+08:00", direction="short")
        short_bars = make_closed_bars([97.0, 94.0, 92.0, 99.0], direction="short")
        short_exit = tce.simulate_atr_chandelier_exit(short_trade, short_bars, 3.0)
        self.assertEqual(short_exit.exit_time, pd.Timestamp("2025-01-01T16:00:00+08:00"))

    def test_trailing_stop_does_not_relax(self) -> None:
        trade = make_internal_trade(entry="2025-01-01T00:00:00+08:00", direction="long")
        bars = make_closed_bars([103.0, 106.0, 105.0, 108.0, 101.0])

        exit_result = tce.simulate_atr_chandelier_exit(trade, bars, 3.0, return_debug_path=True)

        path = exit_result.debug_path or []
        self.assertTrue(all(path[index] >= path[index - 1] for index in range(1, len(path))))

    def test_swing_trailing_uses_previous_level(self) -> None:
        trade = make_internal_trade(entry="2025-01-01T00:00:00+08:00", direction="long")
        bars = make_closed_bars([100.0 + index for index in range(12)])
        bars["low"] = [95.0] * 10 + [94.0, 120.0]
        bars["close"] = [100.0] * 12

        exit_result = tce.simulate_swing_trailing_exit(trade, bars, swing_bars=10)

        self.assertEqual(exit_result.exit_time, pd.Timestamp("2025-01-02T20:00:00+08:00"))

    def test_oracle_hold_to_trend_end_marked_oracle_true(self) -> None:
        trade = make_internal_trade(exit_time="2025-01-01T08:00:00+08:00")
        trades = pd.DataFrame([trade])
        segments_by_symbol = tce.build_segments_by_symbol(make_segments(), "4h")
        funding_histories = {tce.symbol_to_inst_id(SYMBOL): pd.DataFrame(columns=["funding_time_utc", "funding_rate"])}

        exit_trades = tce.build_exit_variant_trades(
            trades,
            segments_by_symbol,
            "4h",
            "Asia/Shanghai",
            funding_histories,
            Path("/missing.db"),
            bars_by_symbol={SYMBOL: make_minute_bars(minutes=2880)},
        )

        oracle = exit_trades[exit_trades["exit_variant"] == tce.ORACLE_VARIANT].iloc[0]
        self.assertTrue(bool(oracle["oracle"]))

    def test_funding_alignment_is_inclusive(self) -> None:
        inst_id = tce.symbol_to_inst_id(SYMBOL)
        funding = pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(
                    [
                        "2024-12-31T16:00:00Z",
                        "2025-01-01T00:00:00Z",
                        "2025-01-01T08:00:00Z",
                    ],
                    utc=True,
                ),
                "funding_rate": [0.01, 0.02, 0.03],
            }
        )

        result = tce.funding_for_trade(
            inst_id,
            "long",
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-01T08:00:00Z"),
            100.0,
            {inst_id: funding},
        )

        self.assertEqual(result["funding_events_count"], 2)
        self.assertAlmostEqual(result["funding_pnl"], -5.0)

    def test_concentration_calculation(self) -> None:
        trades = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "BBB", "CCC"],
                "funding_adjusted_pnl": [5.0, 5.0, 5.0, -1.0],
            }
        )

        self.assertAlmostEqual(tce.largest_symbol_pnl_share(trades), 10.0 / 15.0)
        self.assertAlmostEqual(tce.top_5pct_trade_pnl_contribution(trades), 5.0 / 14.0)

    def test_stable_like_gate_true_case(self) -> None:
        by_split = pd.DataFrame(
            [
                {
                    "exit_variant": "variant",
                    "oracle": False,
                    "split": split,
                    "trade_count": 10,
                    "no_cost_pnl": 1.0,
                    "cost_aware_pnl": 1.0,
                    "funding_adjusted_pnl": 1.0,
                    "largest_symbol_pnl_share": 0.5,
                    "top_5pct_trade_pnl_contribution": 0.5,
                }
                for split in tce.SPLITS
            ]
        )
        by_variant = pd.DataFrame(
            [
                {"exit_variant": "original_exit", "oracle": False, "avg_captured_fraction": 0.10, "early_exit_share": 0.90},
                {"exit_variant": "variant", "oracle": False, "avg_captured_fraction": 0.20, "early_exit_share": 0.70},
            ]
        )

        rejected, candidates = tce.evaluate_stable_like_gates(by_split, by_variant, funding_data_complete=True)

        self.assertTrue(candidates)
        self.assertTrue(bool(rejected[rejected["exit_variant"] == "variant"].iloc[0]["stable_like"]))

    def test_stable_like_gate_false_case(self) -> None:
        by_split = pd.DataFrame(
            [
                {
                    "exit_variant": "variant",
                    "oracle": False,
                    "split": split,
                    "trade_count": 9,
                    "no_cost_pnl": -1.0 if split == "oos_ext" else 1.0,
                    "cost_aware_pnl": -1.0 if split == "oos_ext" else 1.0,
                    "funding_adjusted_pnl": -1.0 if split == "oos_ext" else 1.0,
                    "largest_symbol_pnl_share": 0.9,
                    "top_5pct_trade_pnl_contribution": 0.9,
                }
                for split in tce.SPLITS
            ]
        )
        by_variant = pd.DataFrame(
            [
                {"exit_variant": "original_exit", "oracle": False, "avg_captured_fraction": 0.10, "early_exit_share": 0.90},
                {"exit_variant": "variant", "oracle": False, "avg_captured_fraction": 0.11, "early_exit_share": 0.89},
            ]
        )

        rejected, candidates = tce.evaluate_stable_like_gates(by_split, by_variant, funding_data_complete=True)

        self.assertFalse(candidates)
        self.assertFalse(bool(rejected[rejected["exit_variant"] == "variant"].iloc[0]["stable_like"]))

    def test_output_csv_json_md_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trend_map_dir = root / "trend_map"
            trend_map_dir.mkdir()
            make_segments().drop(columns=["start_ts", "end_ts"]).to_csv(trend_map_dir / "trend_segments.csv", index=False)
            pd.DataFrame({"trade_id": ["x"]}).to_csv(trend_map_dir / "legacy_strategy_trend_coverage.csv", index=False)
            (trend_map_dir / "trend_opportunity_summary.json").write_text(
                json.dumps(
                    {
                        "enough_trend_opportunities": True,
                        "trend_opportunities_are_diversified": True,
                        "recommended_next_research_direction": "trend_exit_convexity_research",
                        "data_quality": {"funding_data_complete": True, "all_symbols_complete": True},
                        "legacy_analysis": {"early_exit_share": 0.9, "avg_captured_fraction": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            (trend_map_dir / "data_quality.json").write_text(json.dumps({"all_symbols_complete": True}), encoding="utf-8")

            trend_v3_dir = root / "trend_v3"
            for split in tce.SPLITS:
                split_dir = trend_v3_dir / split
                split_dir.mkdir(parents=True)
                trade = make_internal_trade(split=split)
                pd.DataFrame(
                    [
                        {
                            "policy_name": "policy",
                            "symbol": SYMBOL,
                            "direction": "long",
                            "entry_time": trade["entry_time"],
                            "entry_price": 100.0,
                            "exit_time": trade["exit_time"],
                            "exit_price": 102.0,
                            "holding_minutes": trade["holding_minutes"],
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

            funding_dir = root / "funding"
            funding_dir.mkdir()
            pd.DataFrame(
                {
                    "inst_id": [tce.symbol_to_inst_id(SYMBOL)],
                    "funding_time_utc": ["2025-01-01T00:00:00Z"],
                    "funding_rate": [0.0],
                }
            ).to_csv(funding_dir / f"{tce.symbol_to_inst_id(SYMBOL)}_funding_2025.csv", index=False)

            outputs = tce.run_research(
                trend_map_dir=trend_map_dir,
                trend_v3_dir=trend_v3_dir,
                vsvcb_dir=root / "missing_vsvcb",
                csrb_dir=root / "missing_csrb",
                funding_dir=funding_dir,
                output_dir=root / "out",
                timezone_name="Asia/Shanghai",
                primary_timeframe="4h",
                data_check_strict=True,
                logger=logging.getLogger("test_tce"),
                database_path=root / "missing.db",
                bars_by_symbol={SYMBOL: make_minute_bars(minutes=4320)},
            )

            self.assertFalse(outputs.exit_trades.empty)
            for filename in tce.REQUIRED_OUTPUT_FILES:
                self.assertTrue((outputs.output_dir / filename).exists(), filename)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-trend-exit-convexity:", makefile)
        self.assertIn("scripts/research_trend_capture_exit_convexity.py", makefile)


if __name__ == "__main__":
    unittest.main()
