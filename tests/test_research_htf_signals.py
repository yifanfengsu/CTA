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

import compare_htf_signal_research as compare_mod
import research_htf_signals as htf_mod


def make_1m_bars(minutes: int = 1440, start: str = "2025-01-01T00:00:00+08:00") -> pd.DataFrame:
    """Build deterministic upward 1m OHLCV bars."""

    start_dt = pd.Timestamp(start)
    records = []
    previous_close = 100.0
    for index in range(minutes):
        wave = 0.15 if index % 17 == 0 else 0.0
        close = 100.0 + index * 0.02 + wave
        open_price = previous_close
        high = max(open_price, close) + 0.2
        low = min(open_price, close) - 0.2
        records.append(
            {
                "datetime": (start_dt + pd.Timedelta(minutes=index)).isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100.0 + (index % 20),
            }
        )
        previous_close = close
    return pd.DataFrame(records)


def base_aligned_row(**overrides: object) -> dict[str, object]:
    """Build a minimal aligned 5m/15m/1h row for condition tests."""

    row: dict[str, object] = {
        "datetime": pd.Timestamp("2025-01-01T02:00:00+08:00"),
        "close_1h": 110.0,
        "ema50_1h": 105.0,
        "ema200_1h": 100.0,
        "ema50_slope_1h": 0.2,
        "close_15m": 110.0,
        "ema21_15m": 108.0,
        "ema55_15m": 106.0,
        "rolling_vwap_15m": 107.0,
        "donchian_mid_15m": 105.0,
        "donchian_high_slope_15m": 0.1,
        "donchian_low_slope_15m": 0.1,
        "atr_pct_percentile_15m": 0.5,
        "recent_volatility_30m_percentile_15m": 0.5,
        "directional_recent_return_30m_percentile_long_15m": 0.5,
        "directional_recent_return_30m_percentile_short_15m": 0.5,
        "volume_zscore_30m_percentile_15m": 0.5,
        "body_ratio_percentile_15m": 0.5,
        "low_5m": 108.1,
        "high_5m": 111.0,
        "close_5m": 110.0,
        "ema21_5m": 109.0,
        "atr14_15m": 2.0,
    }
    row.update(overrides)
    return row


def make_outcome_bars(start: str, rows: list[dict[str, float]]) -> pd.DataFrame:
    """Build compact 1m future bars for outcome tests."""

    start_dt = pd.Timestamp(start)
    records = []
    for offset, row in enumerate(rows, start=1):
        records.append(
            {
                "datetime": start_dt + pd.Timedelta(minutes=offset),
                "open": row.get("open", row["close"]),
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": 1.0,
            }
        )
    return pd.DataFrame(records)


def make_signal(direction: str, entry_price: float = 100.0, start: str = "2025-01-01T00:00:00+08:00") -> pd.DataFrame:
    """Build one minimal signal row."""

    entry_dt = pd.Timestamp(start)
    return pd.DataFrame(
        [
            {
                "_signal_dt": entry_dt,
                "_entry_dt": entry_dt,
                "signal_time": entry_dt.isoformat(),
                "entry_time": entry_dt.isoformat(),
                "entry_price": entry_price,
                "direction": direction,
            }
        ]
    )


class ResearchHtfSignalsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_research_htf_signals")
        self.logger.handlers.clear()

    def test_1m_bars_resample_to_5m_15m_1h(self) -> None:
        bars = htf_mod.normalize_1m_bars(make_1m_bars(60), "Asia/Shanghai")
        five = htf_mod.resample_ohlcv(bars, 5)
        fifteen = htf_mod.resample_ohlcv(bars, 15)
        hourly = htf_mod.resample_ohlcv(bars, 60)

        self.assertEqual(len(five.index), 12)
        self.assertEqual(len(fifteen.index), 4)
        self.assertEqual(len(hourly.index), 1)
        self.assertAlmostEqual(float(five.iloc[0]["open"]), float(bars.iloc[0]["open"]))
        self.assertAlmostEqual(float(five.iloc[0]["close"]), float(bars.iloc[4]["close"]))
        self.assertAlmostEqual(float(five.iloc[0]["high"]), float(bars.iloc[:5]["high"].max()))
        self.assertAlmostEqual(float(five.iloc[0]["volume"]), float(bars.iloc[:5]["volume"].sum()))

        partial = htf_mod.resample_ohlcv(htf_mod.normalize_1m_bars(make_1m_bars(61), "Asia/Shanghai"), 60)
        self.assertEqual(len(partial.index), 1)
        self.assertEqual(pd.Timestamp(partial.iloc[-1]["datetime"]), pd.Timestamp("2025-01-01T00:59:00+08:00"))

    def test_indicators_generate_ema_atr_donchian_vwap(self) -> None:
        bars = htf_mod.resample_ohlcv(htf_mod.normalize_1m_bars(make_1m_bars(300), "Asia/Shanghai"), 5)
        indicators = htf_mod.compute_indicators(bars, 5)

        for column in ["ema21", "ema50", "ema55", "ema200", "atr14", "atr_pct", "donchian_high", "donchian_low", "donchian_mid", "rolling_vwap"]:
            self.assertIn(column, indicators.columns)
            self.assertFalse(indicators[column].dropna().empty, column)

    def test_1h_ema_regime_long_and_short(self) -> None:
        long_df = htf_mod.add_policy_conditions(pd.DataFrame([base_aligned_row()]))
        short_df = htf_mod.add_policy_conditions(
            pd.DataFrame(
                [
                    base_aligned_row(
                        close_1h=90.0,
                        ema50_1h=95.0,
                        ema200_1h=100.0,
                        ema50_slope_1h=-0.2,
                    )
                ]
            )
        )

        self.assertTrue(bool(long_df.iloc[0]["regime_long_1h"]))
        self.assertTrue(bool(short_df.iloc[0]["regime_short_1h"]))

    def test_15m_ema_structure_long_and_short(self) -> None:
        long_df = htf_mod.add_policy_conditions(pd.DataFrame([base_aligned_row()]))
        short_df = htf_mod.add_policy_conditions(
            pd.DataFrame([base_aligned_row(close_15m=90.0, ema21_15m=92.0, ema55_15m=95.0)])
        )

        self.assertTrue(bool(long_df.iloc[0]["ema_structure_long_15m"]))
        self.assertTrue(bool(short_df.iloc[0]["ema_structure_short_15m"]))

    def test_vol_cap_filters_high_atr_pct(self) -> None:
        filtered = htf_mod.add_policy_conditions(
            pd.DataFrame([base_aligned_row(atr_pct_percentile_15m=0.9, recent_volatility_30m_percentile_15m=0.4)])
        )
        allowed = htf_mod.add_policy_conditions(
            pd.DataFrame([base_aligned_row(atr_pct_percentile_15m=0.7, recent_volatility_30m_percentile_15m=0.4)])
        )

        self.assertFalse(bool(filtered.iloc[0]["vol_cap_80_15m"]))
        self.assertTrue(bool(allowed.iloc[0]["vol_cap_80_15m"]))

    def test_no_overextension_filters_high_directional_return(self) -> None:
        filtered = htf_mod.add_policy_conditions(
            pd.DataFrame([base_aligned_row(directional_recent_return_30m_percentile_long_15m=0.9)])
        )
        allowed = htf_mod.add_policy_conditions(
            pd.DataFrame([base_aligned_row(directional_recent_return_30m_percentile_long_15m=0.7)])
        )

        self.assertFalse(bool(filtered.iloc[0]["no_overextension_long_15m"]))
        self.assertTrue(bool(allowed.iloc[0]["no_overextension_long_15m"]))

    def test_short_future_return_uses_inverse_direction_formula(self) -> None:
        bars = make_outcome_bars(
            "2025-01-01T00:00:00+08:00",
            [
                {"high": 101.0, "low": 95.0, "close": 95.0},
                {"high": 102.0, "low": 80.0, "close": 80.0},
            ],
        )
        result = htf_mod.compute_signal_outcomes(make_signal("short"), bars, [2], [])

        self.assertAlmostEqual(float(result.iloc[0]["future_return_2m"]), 100.0 / 80.0 - 1.0)

    def test_long_future_return_uses_close_over_entry_formula(self) -> None:
        bars = make_outcome_bars(
            "2025-01-01T00:00:00+08:00",
            [
                {"high": 110.0, "low": 99.0, "close": 110.0},
                {"high": 120.0, "low": 98.0, "close": 120.0},
            ],
        )
        result = htf_mod.compute_signal_outcomes(make_signal("long"), bars, [2], [])

        self.assertAlmostEqual(float(result.iloc[0]["future_return_2m"]), 120.0 / 100.0 - 1.0)

    def test_short_mfe_mae_use_inverse_direction_formula(self) -> None:
        bars = make_outcome_bars(
            "2025-01-01T00:00:00+08:00",
            [
                {"high": 110.0, "low": 90.0, "close": 100.0},
                {"high": 105.0, "low": 80.0, "close": 95.0},
            ],
        )
        result = htf_mod.compute_signal_outcomes(make_signal("short"), bars, [2], [])

        self.assertAlmostEqual(float(result.iloc[0]["mfe_2m"]), 100.0 / 80.0 - 1.0)
        self.assertAlmostEqual(float(result.iloc[0]["mae_2m"]), 1.0 - 100.0 / 110.0)

    def test_long_mfe_mae_use_long_direction_formula(self) -> None:
        bars = make_outcome_bars(
            "2025-01-01T00:00:00+08:00",
            [
                {"high": 120.0, "low": 95.0, "close": 110.0},
                {"high": 115.0, "low": 90.0, "close": 105.0},
            ],
        )
        result = htf_mod.compute_signal_outcomes(make_signal("long"), bars, [2], [])

        self.assertAlmostEqual(float(result.iloc[0]["mfe_2m"]), 120.0 / 100.0 - 1.0)
        self.assertAlmostEqual(float(result.iloc[0]["mae_2m"]), 1.0 - 90.0 / 100.0)

    def test_resample_alignment_uses_only_completed_15m_and_1h_bars(self) -> None:
        bars = htf_mod.normalize_1m_bars(make_1m_bars(130), "Asia/Shanghai")
        history_range = htf_mod.resolve_split_range(
            "train",
            "2025-01-01T00:00:00+08:00",
            "2025-01-01T02:10:00+08:00",
            "Asia/Shanghai",
        )
        aligned = htf_mod.align_indicator_frames(htf_mod.build_indicator_frames(htf_mod.build_timeframes(bars)), history_range)
        row = aligned[aligned["datetime"] == pd.Timestamp("2025-01-01T01:04:00+08:00")].iloc[0]

        self.assertEqual(pd.Timestamp(row["used_15m_bar_time"]), pd.Timestamp("2025-01-01T00:59:00+08:00"))
        self.assertEqual(pd.Timestamp(row["used_1h_bar_time"]), pd.Timestamp("2025-01-01T00:59:00+08:00"))
        self.assertLessEqual(pd.Timestamp(row["used_15m_bar_time"]), pd.Timestamp(row["datetime"]))
        self.assertLessEqual(pd.Timestamp(row["used_1h_bar_time"]), pd.Timestamp(row["datetime"]))

    def test_rolling_percentile_excludes_current_and_future_values(self) -> None:
        first = htf_mod.rolling_percentile(pd.Series([4.0, 3.0, 2.0, 1.0, 5.0]), window=3)
        with_future_outlier = htf_mod.rolling_percentile(pd.Series([4.0, 3.0, 2.0, 1.0, 999.0]), window=3)

        self.assertAlmostEqual(float(first.iloc[3]), 0.0)
        self.assertAlmostEqual(float(first.iloc[3]), float(with_future_outlier.iloc[3]))

    def test_5m_pullback_reclaim_identifies_long_example(self) -> None:
        rows = [
            base_aligned_row(
                datetime=pd.Timestamp("2025-01-01T02:00:00+08:00"),
                low_5m=108.1,
                close_5m=108.5,
                ema21_5m=109.0,
            ),
            base_aligned_row(
                datetime=pd.Timestamp("2025-01-01T02:05:00+08:00"),
                low_5m=108.4,
                close_5m=110.0,
                ema21_5m=109.0,
            ),
        ]
        result = htf_mod.add_policy_conditions(pd.DataFrame(rows))

        self.assertTrue(bool(result.iloc[1]["pullback_recent_long_5m"]))
        self.assertTrue(bool(result.iloc[1]["reclaim_long_5m"]))
        self.assertTrue(htf_mod.policy_condition(result.iloc[1], "htf_1h_15m_structure_5m_pullback_reclaim", "long"))

    def test_pullback_reclaim_signal_enters_on_next_5m_close(self) -> None:
        rows = [
            base_aligned_row(
                datetime=pd.Timestamp("2025-01-01T02:00:00+08:00"),
                low_5m=108.1,
                close_5m=108.5,
                ema21_5m=109.0,
            ),
            base_aligned_row(
                datetime=pd.Timestamp("2025-01-01T02:05:00+08:00"),
                low_5m=108.4,
                close_5m=110.0,
                ema21_5m=109.0,
            ),
            base_aligned_row(
                datetime=pd.Timestamp("2025-01-01T02:10:00+08:00"),
                low_5m=109.5,
                close_5m=111.0,
                ema21_5m=109.5,
            ),
        ]
        aligned = htf_mod.add_policy_conditions(pd.DataFrame(rows))
        signals = htf_mod.apply_cooldown_and_build_signals(aligned, "BTCUSDT_SWAP_OKX.GLOBAL", 0, None)
        signal = signals[signals["policy_name"] == "htf_1h_15m_structure_5m_pullback_reclaim"].iloc[0]

        self.assertEqual(pd.Timestamp(signal["signal_time"]), pd.Timestamp("2025-01-01T02:05:00+08:00"))
        self.assertEqual(pd.Timestamp(signal["entry_time"]), pd.Timestamp("2025-01-01T02:10:00+08:00"))
        self.assertAlmostEqual(float(signal["entry_price"]), 111.0)
        self.assertEqual(int(signal["entry_delay_bars_5m"]), 1)

    def test_bracket_same_bar_stop_and_tp_is_stop_first(self) -> None:
        signal = pd.Series(
            {
                "_signal_dt": pd.Timestamp("2025-01-01T00:00:00+08:00"),
                "entry_price": 100.0,
                "atr_reference": 10.0,
                "direction": "long",
            }
        )
        bars = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp("2025-01-01T00:01:00+08:00"),
                    "open": 100.0,
                    "high": 111.0,
                    "low": 89.0,
                    "close": 100.0,
                    "volume": 1.0,
                }
            ]
        )
        result = htf_mod.simulate_bracket_for_signal(
            signal,
            bars,
            pd.Series(bars["datetime"]),
            horizon_minutes=60,
            stop_atr=1.0,
            tp_atr=1.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.exit_type, "stop")
        self.assertEqual(result.r_multiple, -1.0)

    def test_research_outputs_csv_json_md_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "htf"
            history_range = htf_mod.resolve_split_range(
                "train",
                "2025-01-02",
                "2025-01-03",
                "Asia/Shanghai",
            )
            summary = htf_mod.run_research(
                vt_symbol="BTCUSDT_SWAP_OKX.GLOBAL",
                split="train",
                history_range=history_range,
                output_dir=output_dir,
                horizons=[60, 120],
                stop_grid=[1.5],
                tp_grid=[2.0],
                timezone_name="Asia/Shanghai",
                cooldown_bars_5m=6,
                data_check_strict=False,
                logger=self.logger,
                max_signals=20,
                bars_from_db=False,
                bars_df=make_1m_bars(3 * 24 * 60),
            )

            for filename in [
                "htf_signal_dataset.csv",
                "htf_policy_summary.json",
                "htf_policy_leaderboard.csv",
                "htf_bracket_grid.csv",
                "htf_policy_by_side.csv",
                "htf_policy_by_hour.csv",
                "htf_policy_by_weekday.csv",
                "htf_research_report.md",
                "htf_research_audit.json",
                "data_quality.json",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)

        self.assertEqual(summary["split"], "train")
        self.assertIn("diagnostic_answers", summary)

    def test_compare_htf_signal_research_marks_negative_policy_not_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_values = {"train": -0.003, "validation": -0.002, "oos": -0.001}
            split_dirs: dict[str, Path] = {}
            for split, value in split_values.items():
                directory = root / split
                directory.mkdir(parents=True)
                split_dirs[split] = directory
                pd.DataFrame(
                    [
                        {
                            "policy_name": "negative_policy",
                            "signal_count": 20,
                            "median_future_return_120m": value,
                            "best_expectancy_r": 0.1,
                            "positive_rate_120m": 0.4,
                        }
                    ]
                ).to_csv(directory / "htf_policy_leaderboard.csv", index=False)

            output_dir = root / "compare"
            summary = compare_mod.run_compare(
                split_dirs["train"],
                split_dirs["validation"],
                split_dirs["oos"],
                output_dir,
            )
            compare_df = pd.read_csv(output_dir / "htf_compare_leaderboard.csv")

        row = compare_df[compare_df["policy_name"] == "negative_policy"].iloc[0]
        self.assertFalse(bool(row["stable_candidate"]))
        self.assertTrue(summary["no_stable_htf_policy"])


if __name__ == "__main__":
    unittest.main()
