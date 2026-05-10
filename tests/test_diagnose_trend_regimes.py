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

import diagnose_trend_regimes as regime_mod


SYMBOL = "AAAUSDT_SWAP_OKX.GLOBAL"


def make_1m_bars(symbol: str, minutes: int, start: str = "2025-01-01T00:00:00+08:00") -> pd.DataFrame:
    rows = []
    start_dt = pd.Timestamp(start)
    previous_close = 100.0
    for index in range(minutes):
        close = 100.0 + index * 0.01
        rows.append(
            {
                "vt_symbol": symbol,
                "datetime": (start_dt + pd.Timedelta(minutes=index)).isoformat(),
                "open": previous_close,
                "high": max(previous_close, close) + 0.1,
                "low": min(previous_close, close) - 0.1,
                "close": close,
                "volume": 10.0 + index % 5,
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def regime_row(**overrides: object) -> pd.Series:
    base = {
        "ema_spread_pct": 0.0,
        "ema50_slope": 0.0,
        "ema200_slope": 0.0,
        "close_distance_to_ema50": 0.0,
        "close_distance_to_ema200": 0.0,
        "trend_efficiency": 0.05,
        "adx_or_proxy": 5.0,
        "atr_percentile": 0.5,
        "realized_volatility_percentile": 0.5,
        "donchian_width_percentile": 0.5,
    }
    base.update(overrides)
    return pd.Series(base)


class DiagnoseTrendRegimesTest(unittest.TestCase):
    def test_resample_4h_and_1d_uses_only_closed_bars(self) -> None:
        incomplete_4h = make_1m_bars(SYMBOL, 239)
        complete_4h = make_1m_bars(SYMBOL, 240)
        incomplete_1d = make_1m_bars(SYMBOL, 1439)
        complete_1d = make_1m_bars(SYMBOL, 1440)

        self.assertEqual(len(regime_mod.resample_ohlcv(incomplete_4h, 240).index), 0)
        four_hour = regime_mod.resample_ohlcv(complete_4h, 240)
        self.assertEqual(pd.Timestamp(four_hour.iloc[0]["datetime"]), pd.Timestamp("2025-01-01T03:59:00+08:00"))
        self.assertEqual(len(regime_mod.resample_ohlcv(incomplete_1d, 1440).index), 0)
        daily = regime_mod.resample_ohlcv(complete_1d, 1440)
        self.assertEqual(pd.Timestamp(daily.iloc[0]["datetime"]), pd.Timestamp("2025-01-01T23:59:00+08:00"))

    def test_trend_efficiency_calculation(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        efficiency = regime_mod.compute_trend_efficiency(close, 3)
        self.assertAlmostEqual(float(efficiency.iloc[-1]), 1.0)

        choppy_close = pd.Series([100.0, 101.0, 100.0, 101.0])
        choppy_efficiency = regime_mod.compute_trend_efficiency(choppy_close, 3)
        self.assertAlmostEqual(float(choppy_efficiency.iloc[-1]), 1.0 / 3.0)

    def test_ema_spread_and_slope_calculation(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=260, freq="4h", tz="Asia/Shanghai"),
                "open": [100.0 + i for i in range(260)],
                "high": [101.0 + i for i in range(260)],
                "low": [99.0 + i for i in range(260)],
                "close": [100.0 + i for i in range(260)],
                "volume": [1.0] * 260,
            }
        )
        indicators = regime_mod.compute_regime_indicators(frame, "4h", [20, 55, 100])
        row = indicators.iloc[-1]

        self.assertAlmostEqual(float(row["ema_spread_pct"]), float((row["ema50"] - row["ema200"]) / row["close"]))
        self.assertAlmostEqual(
            float(row["ema50_slope"]),
            float(indicators.iloc[-1]["ema50"] - indicators.iloc[-2]["ema50"]),
        )
        self.assertGreater(float(row["ema_spread_pct"]), 0.0)
        self.assertGreater(float(row["ema50_slope"]), 0.0)

    def test_regime_label_identifies_strong_uptrend(self) -> None:
        label = regime_mod.classify_regime_row(
            regime_row(
                ema_spread_pct=0.03,
                ema50_slope=1.0,
                ema200_slope=0.5,
                close_distance_to_ema50=0.02,
                close_distance_to_ema200=0.05,
                trend_efficiency=0.75,
                adx_or_proxy=35.0,
            ),
            "4h",
            [20, 55, 100],
        )

        self.assertEqual(label, "strong_uptrend")

    def test_regime_label_identifies_choppy(self) -> None:
        label = regime_mod.classify_regime_row(regime_row(trend_efficiency=0.08), "4h", [20, 55, 100])

        self.assertEqual(label, "choppy")

    def test_regime_label_identifies_high_vol_choppy(self) -> None:
        label = regime_mod.classify_regime_row(
            regime_row(trend_efficiency=0.08, atr_percentile=0.92, realized_volatility_percentile=0.88),
            "4h",
            [20, 55, 100],
        )

        self.assertEqual(label, "high_vol_choppy")

    def test_trade_regime_attribution_aligns_entry_time(self) -> None:
        regime_dataset = pd.DataFrame(
            {
                "symbol": [SYMBOL, SYMBOL],
                "timeframe": ["4h", "4h"],
                "datetime": [
                    pd.Timestamp("2025-01-01T03:59:00+08:00"),
                    pd.Timestamp("2025-01-01T07:59:00+08:00"),
                ],
                "regime_label": ["strong_uptrend", "choppy"],
                "trend_efficiency": [0.6, 0.1],
                "ema_spread_pct": [0.02, 0.0],
                "atr_pct": [0.01, 0.02],
                "adx_or_proxy": [30.0, 5.0],
            }
        )
        trades = pd.DataFrame(
            {
                "split": ["oos_ext"],
                "policy_name": ["v3_4h_ema_50_200_atr4"],
                "symbol": [SYMBOL],
                "direction": ["long"],
                "entry_time": ["2025-01-01T04:00:00+08:00"],
                "exit_time": ["2025-01-01T08:00:00+08:00"],
                "net_pnl": [1.0],
                "no_cost_pnl": [1.1],
                "timeframe": ["4h"],
            }
        )

        attributed = regime_mod.align_trades_to_regimes(trades, regime_dataset, ["4h"])

        self.assertEqual(attributed.iloc[0]["regime_at_entry"], "strong_uptrend")
        self.assertTrue(bool(attributed.iloc[0]["is_regime_aligned"]))
        self.assertAlmostEqual(float(attributed.iloc[0]["trend_efficiency"]), 0.6)

    def test_v3_1_recommendation_true_case(self) -> None:
        recommendation = regime_mod.build_v3_1_recommendations(
            {"pct": {"strong_uptrend": 0.10, "strong_downtrend": 0.04}},
            {
                "strong_positive_pnl_share": 0.60,
                "choppy_high_vol_loss_share": 0.70,
                "one_day_ema_strong_outperforms_full": True,
                "family_diagnostics": {"1d_ema": {"strong_no_cost_pnl": 1.0}},
            },
            {},
        )

        self.assertTrue(recommendation["proceed_to_v3_1_research"])
        self.assertFalse(recommendation["strategy_development_allowed"])
        self.assertFalse(recommendation["demo_live_allowed"])
        self.assertIn("1d_ema", recommendation["recommended_policy_families"])

    def test_v3_1_recommendation_false_case(self) -> None:
        recommendation = regime_mod.build_v3_1_recommendations(
            {"pct": {"strong_uptrend": 0.01, "strong_downtrend": 0.01}},
            {
                "strong_positive_pnl_share": 0.20,
                "choppy_high_vol_loss_share": 0.30,
                "one_day_ema_strong_outperforms_full": False,
                "family_diagnostics": {"1d_ema": {"strong_no_cost_pnl": 1.0}},
            },
            {},
        )

        self.assertFalse(recommendation["proceed_to_v3_1_research"])
        self.assertEqual(recommendation["recommended_policy_families"], [])
        self.assertFalse(recommendation["strategy_development_allowed"])
        self.assertFalse(recommendation["demo_live_allowed"])

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "trend_regime_diagnostics"
            outputs = regime_mod.run_diagnostics(
                symbols=[SYMBOL],
                start="2025-01-01",
                end="2025-01-01",
                timezone_name="Asia/Shanghai",
                output_dir=output_dir,
                timeframes=["4h", "1d"],
                windows=[3],
                data_check_strict=True,
                logger=logging.getLogger("test_diagnose_trend_regimes"),
                bars_by_symbol={SYMBOL: make_1m_bars(SYMBOL, 1440)},
                trade_paths={},
            )

            self.assertTrue(outputs.data_quality["all_symbols_complete"])
            for filename in regime_mod.REQUIRED_OUTPUT_FILES:
                self.assertTrue((output_dir / filename).exists(), filename)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("diagnose-trend-regimes:", makefile)


if __name__ == "__main__":
    unittest.main()
