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
import research_trend_health_state_exit as health


SYMBOL = "AAAUSDT_SWAP_OKX.GLOBAL"
INST_ID = tce.symbol_to_inst_id(SYMBOL)


def make_trade(
    *,
    entry: str = "2025-01-01T00:00:00+08:00",
    exit_time: str = "2025-01-01T12:00:00+08:00",
    direction: str = "long",
    split: str = "train_ext",
    trade_id: str = "t1",
) -> pd.Series:
    entry_ts = pd.Timestamp(entry)
    exit_ts = pd.Timestamp(exit_time)
    return pd.Series(
        {
            "strategy_source": "trend_v3_extended",
            "policy_or_group": "policy",
            "symbol": SYMBOL,
            "trade_timeframe": "4h",
            "trade_id": trade_id,
            "split": split,
            "direction": direction,
            "entry_time": entry_ts.isoformat(),
            "entry_ts": entry_ts,
            "entry_price": 100.0,
            "exit_time": exit_ts.isoformat(),
            "exit_ts": exit_ts,
            "exit_price": 102.0,
            "holding_minutes": (exit_ts - entry_ts).total_seconds() / 60.0,
            "no_cost_pnl": 2.0,
            "cost_aware_pnl": 1.9,
            "cost_drag": 0.1,
            "pnl_multiplier": 1.0,
            "notional": 100.0,
        }
    )


def make_health_bars(
    closes: list[float],
    *,
    ema: list[float] | None = None,
    volume_ratio: list[float] | None = None,
    open_base: float = 100.0,
) -> pd.DataFrame:
    datetimes = pd.date_range("2025-01-01T04:00:00+08:00", periods=len(closes), freq="4h")
    open_times = pd.date_range("2025-01-01T00:00:00+08:00", periods=len(closes), freq="4h")
    rows = []
    for index, close in enumerate(closes):
        ema_value = ema[index] if ema is not None else close - 10.0
        rows.append(
            {
                "open_time": open_times[index],
                "datetime": datetimes[index],
                "open": open_base + index,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1.0,
                "ema20": ema_value,
                "ema50": ema_value,
                "atr14": 10.0,
                "volume_sma20": 1.0,
                "volume_ratio": volume_ratio[index] if volume_ratio is not None else 1.0,
            }
        )
    return tce.add_time_ns(pd.DataFrame(rows))


class TrendHealthStateExitResearchTest(unittest.TestCase):
    def test_direction_aware_efficiency_long(self) -> None:
        row = pd.Series({"close": 110.0, "ema20": 100.0, "atr14": 5.0})

        self.assertAlmostEqual(health.direction_aware_efficiency(row, "long", "ema20"), 2.0)

    def test_direction_aware_efficiency_short(self) -> None:
        row = pd.Series({"close": 90.0, "ema20": 100.0, "atr14": 5.0})

        self.assertAlmostEqual(health.direction_aware_efficiency(row, "short", "ema20"), 2.0)

    def test_volume_sma_uses_shift_one(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01T00:00:00+08:00", periods=3, freq="4h"),
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [10.0, 20.0, 45.0],
            }
        )

        out = health.add_health_indicators(frame)

        self.assertTrue(pd.isna(out.loc[0, "volume_ratio"]))
        self.assertAlmostEqual(out.loc[1, "volume_ratio"], 2.0)
        self.assertAlmostEqual(out.loc[2, "volume_ratio"], 3.0)

    def test_drawdown_atr_long(self) -> None:
        self.assertAlmostEqual(health.drawdown_atr_from_state("long", 110.0, 120.0, 5.0), 2.0)

    def test_drawdown_atr_short(self) -> None:
        self.assertAlmostEqual(health.drawdown_atr_from_state("short", 90.0, 80.0, 5.0), 2.0)

    def test_health_score_calculation(self) -> None:
        config = health.HealthVariantConfig(name="test")
        row = pd.Series({"close": 110.0, "ema20": 107.0, "atr14": 4.0, "volume_ratio": 0.9})

        state = health.health_state_for_bar(row, direction="long", config=config, best_close=114.0, entry_atr=4.0)

        self.assertTrue(state["efficiency_ok"])
        self.assertTrue(state["energy_ok"])
        self.assertTrue(state["drawdown_ok"])
        self.assertEqual(state["health_score"], 3)

    def test_patience_bars_consecutive_deterioration_exit(self) -> None:
        bars = make_health_bars([100.0, 100.0, 100.0, 100.0], ema=[100.0] * 4, volume_ratio=[0.5] * 4)
        config = health.HealthVariantConfig(name="test", patience_bars=2, max_hold_bars=10)

        result = health.simulate_health_exit(make_trade(), bars, config)

        self.assertEqual(result.exit_reason, "health_deterioration")
        self.assertEqual(result.exit_time, pd.Timestamp("2025-01-01T08:00:00+08:00"))
        self.assertEqual(result.exit_price, 102.0)

    def test_hard_drawdown_exit(self) -> None:
        bars = make_health_bars([120.0, 70.0, 70.0], ema=[100.0, 100.0, 100.0], volume_ratio=[1.0, 1.0, 1.0])
        config = health.HealthVariantConfig(name="test", patience_bars=5, max_hold_bars=10)

        result = health.simulate_health_exit(make_trade(), bars, config)

        self.assertEqual(result.exit_reason, "hard_drawdown")
        self.assertEqual(result.exit_time, pd.Timestamp("2025-01-01T08:00:00+08:00"))

    def test_max_hold_bars_exit(self) -> None:
        bars = make_health_bars([120.0, 121.0, 122.0, 123.0], ema=[100.0] * 4, volume_ratio=[1.0] * 4)
        config = health.HealthVariantConfig(name="test", patience_bars=5, max_hold_bars=2)

        result = health.simulate_health_exit(make_trade(), bars, config)

        self.assertEqual(result.exit_reason, "max_hold_bars")
        self.assertEqual(result.exit_time, pd.Timestamp("2025-01-01T08:00:00+08:00"))

    def test_exit_execution_uses_next_bar_open(self) -> None:
        bars = make_health_bars([100.0, 100.0, 100.0], ema=[100.0] * 3, volume_ratio=[0.5] * 3, open_base=200.0)
        config = health.HealthVariantConfig(name="test", patience_bars=1, max_hold_bars=10)

        result = health.simulate_health_exit(make_trade(), bars, config)

        self.assertEqual(result.exit_time, pd.Timestamp("2025-01-01T04:00:00+08:00"))
        self.assertEqual(result.exit_price, 201.0)

    def test_health_no_energy_variant_ignores_energy_score(self) -> None:
        config = health.HealthVariantConfig(name="no_energy", mode="no_energy")
        row = pd.Series({"close": 99.0, "ema20": 100.0, "atr14": 10.0, "volume_ratio": 2.0})

        state = health.health_state_for_bar(row, direction="long", config=config, best_close=100.0, entry_atr=10.0)

        self.assertEqual(state["health_score"], 1)

    def test_health_energy_confirmed_variant(self) -> None:
        config = health.HealthVariantConfig(name="energy_confirmed", mode="energy_confirmed")

        self.assertTrue(health.health_state_is_unhealthy({"efficiency_ok": False, "energy_ok": False, "drawdown_ok": True, "health_score": 1}, config))
        self.assertFalse(health.health_state_is_unhealthy({"efficiency_ok": False, "energy_ok": True, "drawdown_ok": True, "health_score": 2}, config))

    def test_funding_alignment(self) -> None:
        funding = pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(["2024-12-31T16:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
                "funding_rate": [0.01, 0.02],
            }
        )
        indexes = tce.build_funding_indexes({INST_ID: funding})

        result = health.funding_for_variant(make_trade(entry="2025-01-01T00:00:00+08:00", exit_time="2025-01-01T08:00:00+08:00"), pd.Timestamp("2025-01-01T08:00:00+08:00"), indexes)

        self.assertEqual(result["funding_events_count"], 2)
        self.assertLess(result["funding_pnl"], 0.0)

    def test_concentration_calculation(self) -> None:
        frame = pd.DataFrame(
            [
                {"exit_variant": "v", "oracle": False, "split": "train_ext", "symbol": "A", "funding_adjusted_pnl": 100.0},
                {"exit_variant": "v", "oracle": False, "split": "train_ext", "symbol": "B", "funding_adjusted_pnl": 50.0},
            ]
        )

        out = health.build_concentration_summary(frame)
        all_row = out[out["scope"] == "all_splits"].iloc[0]

        self.assertAlmostEqual(all_row["largest_symbol_pnl_share"], 100.0 / 150.0)

    def test_stable_gate_true_case(self) -> None:
        rows = []
        for variant, oracle in [("original_exit", False), ("health_ema20_core", False)]:
            row = {
                "exit_variant": variant,
                "oracle": oracle,
                "avg_captured_fraction": 0.20 if variant == "original_exit" else 0.40,
                "early_exit_share": 0.80 if variant == "original_exit" else 0.50,
                "largest_symbol_pnl_share": 0.5,
                "top_5pct_trade_pnl_contribution": 0.5,
            }
            for split in health.SPLITS:
                row[f"{split}_trade_count"] = 10
                row[f"{split}_no_cost_pnl"] = 1.0
                row[f"{split}_cost_aware_pnl"] = 1.0
                row[f"{split}_funding_adjusted_pnl"] = 1.0
            rows.append(row)
        concentration = pd.DataFrame(
            [
                {"exit_variant": "original_exit", "oracle": False, "scope": "all_splits", "largest_symbol_pnl_share": 0.5, "top_5pct_trade_pnl_contribution": 0.5},
                {"exit_variant": "health_ema20_core", "oracle": False, "scope": "all_splits", "largest_symbol_pnl_share": 0.5, "top_5pct_trade_pnl_contribution": 0.5},
            ]
        )

        rejected, candidates = health.evaluate_stable_gates(pd.DataFrame(rows), concentration, True)

        candidate = rejected[rejected["exit_variant"] == "health_ema20_core"].iloc[0]
        self.assertTrue(bool(candidate["stable_like"]))
        self.assertTrue(candidates)

    def test_stable_gate_false_case(self) -> None:
        rows = []
        for variant in ["original_exit", "health_ema20_core"]:
            row = {
                "exit_variant": variant,
                "oracle": False,
                "avg_captured_fraction": 0.20,
                "early_exit_share": 0.80,
                "largest_symbol_pnl_share": 1.0,
                "top_5pct_trade_pnl_contribution": 1.0,
            }
            for split in health.SPLITS:
                row[f"{split}_trade_count"] = 1
                row[f"{split}_no_cost_pnl"] = -1.0
                row[f"{split}_cost_aware_pnl"] = -1.0
                row[f"{split}_funding_adjusted_pnl"] = -1.0
            rows.append(row)
        concentration = pd.DataFrame(
            [{"exit_variant": "health_ema20_core", "oracle": False, "scope": "all_splits", "largest_symbol_pnl_share": 1.0, "top_5pct_trade_pnl_contribution": 1.0}]
        )

        rejected, candidates = health.evaluate_stable_gates(pd.DataFrame(rows), concentration, True)

        self.assertFalse(candidates)
        self.assertFalse(bool(rejected[rejected["exit_variant"] == "health_ema20_core"].iloc[0]["stable_like"]))

    def write_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, dict[str, pd.DataFrame]]:
        trend_map = root / "trend_map"
        trend_map.mkdir()
        pd.DataFrame(
            [
                {
                    "trend_segment_id": "seg_1",
                    "symbol": SYMBOL,
                    "timeframe": "4h",
                    "direction": "up",
                    "start_time": "2025-01-01T00:00:00+08:00",
                    "end_time": "2025-01-03T00:00:00+08:00",
                    "duration_bars": 12,
                    "abs_trend_return": 0.2,
                }
            ]
        ).to_csv(trend_map / "trend_segments.csv", index=False)
        (trend_map / "trend_opportunity_summary.json").write_text(json.dumps({"enough_trend_opportunities": True, "legacy_analysis": {"main_failure_mode": "entered_middle_or_late"}}), encoding="utf-8")
        (trend_map / "data_quality.json").write_text(json.dumps({"all_symbols_complete": True}), encoding="utf-8")

        trend_v3 = root / "trend_v3"
        for split in health.SPLITS:
            split_dir = trend_v3 / split
            split_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "policy_name": "policy",
                        "symbol": SYMBOL,
                        "direction": "long",
                        "entry_time": "2025-01-01T00:00:00+08:00",
                        "entry_price": 100.0,
                        "exit_time": "2025-01-01T12:00:00+08:00",
                        "exit_price": 103.0,
                        "holding_minutes": 720,
                        "volume": 1.0,
                        "contract_size": 1.0,
                        "no_cost_pnl": 3.0,
                        "net_pnl": 2.9,
                        "fee": 0.05,
                        "slippage": 0.05,
                        "timeframe": "4h",
                    }
                ]
            ).to_csv(split_dir / "trend_v3_trades.csv", index=False)

        capture = root / "capture"
        capture.mkdir()
        pd.DataFrame([{"trade_id": "t1", "captured_fraction_of_segment": 0.1}]).to_csv(capture / "trend_capture_diagnostics.csv", index=False)
        pd.DataFrame([{"exit_variant": "atr_chandelier_3x", "oracle": False, "avg_captured_fraction": 0.3}]).to_csv(capture / "exit_variant_summary.csv", index=False)
        pd.DataFrame([{"exit_variant": "atr_chandelier_3x"}]).to_csv(capture / "exit_variant_trades.csv", index=False)

        funding = root / "funding"
        funding.mkdir()
        pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(["2024-12-31T16:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T08:00:00Z"], utc=True),
                "funding_rate": [0.0, 0.0, 0.0],
            }
        ).to_csv(funding / f"{INST_ID}_funding_test.csv", index=False)

        bars = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01T00:00:00+08:00", periods=3 * 24 * 60, freq="min"),
                "open": [100.0 + index * 0.001 for index in range(3 * 24 * 60)],
                "high": [100.01 + index * 0.001 for index in range(3 * 24 * 60)],
                "low": [99.99 + index * 0.001 for index in range(3 * 24 * 60)],
                "close": [100.0 + index * 0.001 for index in range(3 * 24 * 60)],
                "volume": [1.0 for _ in range(3 * 24 * 60)],
            }
        )
        return trend_map, trend_v3, capture, funding, {SYMBOL: bars}

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trend_map, trend_v3, capture, funding, bars_by_symbol = self.write_fixture(root)
            output_dir = root / "out"

            outputs = health.run_research(
                trend_map_dir=trend_map,
                trend_v3_dir=trend_v3,
                funding_dir=funding,
                output_dir=output_dir,
                symbols=[SYMBOL],
                start="2025-01-01",
                end="2025-01-03",
                timezone_name="Asia/Shanghai",
                timeframes=["4h"],
                data_check_strict=True,
                logger=logging.getLogger("test_health_exit"),
                database_path=root / "missing.db",
                capture_exit_dir=capture,
                bars_by_symbol=bars_by_symbol,
            )

            self.assertTrue(outputs.output_dir.exists())
            for filename in health.REQUIRED_OUTPUT_FILES:
                self.assertTrue((output_dir / filename).exists(), filename)
            summary = json.loads((output_dir / "health_exit_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["strategy_development_allowed"])
            self.assertFalse(summary["demo_live_allowed"])

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-trend-health-exit:", makefile)
        self.assertIn("scripts/research_trend_health_state_exit.py", makefile)


if __name__ == "__main__":
    unittest.main()
