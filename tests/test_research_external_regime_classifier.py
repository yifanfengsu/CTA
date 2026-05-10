from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import research_external_regime_classifier as research_mod


SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]


def make_daily(symbol_index: int = 0, *, days: int = 260, start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=days, freq="1D", tz="Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59)
    base = 100.0 + symbol_index * 10.0
    rows = []
    for i, dt in enumerate(dates):
        close = base + i * (1.0 + symbol_index * 0.05)
        rows.append(
            {
                "datetime": dt,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0 + i,
            }
        )
    return pd.DataFrame(rows)


def make_funding_features(days: int = 260, start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=days, freq="1D", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "datetime": dates,
            "average_funding_rate": [0.0001] * days,
            "median_funding_rate": [0.0001] * days,
            "funding_dispersion": [0.00001] * days,
            "positive_funding_symbol_count": [5] * days,
            "negative_funding_symbol_count": [0] * days,
            "extreme_positive_funding_count": [0] * days,
            "extreme_negative_funding_count": [0] * days,
            "funding_trend_7d": [0.0] * days,
            "funding_trend_30d": [0.0] * days,
        }
    )


def make_feature_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, start in [
        ("train_ext", "2023-01-01"),
        ("validation_ext", "2024-07-01"),
        ("oos_ext", "2025-07-01"),
    ]:
        for i, dt in enumerate(pd.date_range(start=start, periods=20, freq="1D", tz="Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59)):
            strong = i >= 14
            row = {column: 0.0 for column in research_mod.REGIME_FEATURE_COLUMNS}
            row.update(
                {
                    "datetime": dt,
                    "split": split,
                    "number_of_symbols_above_ema50_1d": 5 if strong else 2,
                    "number_of_symbols_above_ema200_1d": 4 if strong else 2,
                    "ema50_ema200_breadth": 4 if strong else 2,
                    "average_ema_spread_pct": 0.05 if strong else 0.0,
                    "median_ema_spread_pct": 0.05 if strong else 0.0,
                    "strong_trend_symbol_count": 3 if strong else 0,
                    "trend_efficiency_mean": 0.80 if strong else 0.20,
                    "trend_efficiency_median": 0.80 if strong else 0.20,
                    "trend_efficiency_dispersion": 0.05,
                    "average_pairwise_correlation_20d": 0.80 if strong else 0.20,
                    "average_pairwise_correlation_60d": 0.75 if strong else 0.20,
                    "return_dispersion_20d": 0.03 if strong else 0.20,
                    "return_dispersion_60d": 0.03 if strong else 0.20,
                    "market_breadth_return_20d": 5 if strong else 1,
                    "market_breadth_return_60d": 5 if strong else 1,
                    "average_atr_pct": 0.03 if strong else 0.10,
                    "median_atr_pct": 0.03 if strong else 0.10,
                    "atr_pct_dispersion": 0.01,
                    "realized_volatility_mean": 0.02 if strong else 0.10,
                    "realized_volatility_dispersion": 0.01,
                    "high_vol_symbol_count": 0 if strong else 3,
                    "low_vol_symbol_count": 2 if strong else 0,
                    "average_drawdown_from_60d_high": -0.02,
                    "max_symbol_drawdown_from_60d_high": -0.05,
                    "rebound_breadth": 3,
                    "symbols_near_60d_high_count": 4 if strong else 1,
                    "average_funding_rate": 0.0001 if strong else 0.0006,
                    "median_funding_rate": 0.0001 if strong else 0.0006,
                    "funding_dispersion": 0.00001,
                    "positive_funding_symbol_count": 5,
                    "negative_funding_symbol_count": 0,
                    "extreme_positive_funding_count": 0 if strong else 3,
                    "extreme_negative_funding_count": 0,
                    "funding_trend_7d": 0.0,
                    "funding_trend_30d": 0.0,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_attribution(stable: bool = True, concentrated: bool = False) -> pd.DataFrame:
    rows = []
    for split in ("train_ext", "validation_ext", "oos_ext"):
        for i in range(12):
            symbol = SYMBOLS[0] if concentrated else SYMBOLS[i % len(SYMBOLS)]
            pnl = 1.0 if stable else (-1.0 if split == "oos_ext" else 1.0)
            rows.append(
                {
                    "split": split,
                    "policy_name": "policy_a",
                    "symbol": symbol,
                    "direction": "long",
                    "entry_time": pd.Timestamp("2025-01-01", tz="Asia/Shanghai"),
                    "exit_time": pd.Timestamp("2025-01-02", tz="Asia/Shanghai"),
                    "net_pnl": pnl,
                    "no_cost_net_pnl": pnl,
                    "funding_adjusted_net_pnl": pnl,
                    "regime_label": "trend_friendly",
                    "regime_features_snapshot": "{}",
                    "is_trend_friendly": True,
                    "is_trend_hostile": False,
                    "is_funding_overheated": False,
                    "is_high_vol_chop": False,
                    "is_broad_uptrend": True,
                    "is_broad_downtrend": False,
                    "is_narrow_single_symbol_trend": False,
                    "is_funding_supportive": True,
                    "is_compression": False,
                }
            )
    return pd.DataFrame(rows)


def make_top_concentrated_attribution() -> pd.DataFrame:
    attribution = make_attribution(stable=True, concentrated=False)
    oos_mask = attribution["split"] == "oos_ext"
    oos_indices = list(attribution[oos_mask].index)
    attribution.loc[oos_indices, ["net_pnl", "no_cost_net_pnl", "funding_adjusted_net_pnl"]] = -1.0
    attribution.loc[oos_indices[0], ["net_pnl", "no_cost_net_pnl", "funding_adjusted_net_pnl"]] = 100.0
    return attribution


class ResearchExternalRegimeClassifierTest(unittest.TestCase):
    def test_feature_dataset_builds(self) -> None:
        symbol_daily = {symbol: make_daily(index) for index, symbol in enumerate(SYMBOLS)}
        dataset = research_mod.build_external_regime_feature_dataset(symbol_daily, make_funding_features())

        self.assertIn("trend_efficiency_mean", dataset.columns)
        self.assertIn("average_pairwise_correlation_20d", dataset.columns)
        self.assertGreater(len(dataset), 0)

    def test_funding_features_merge(self) -> None:
        symbol_daily = {symbol: make_daily(index) for index, symbol in enumerate(SYMBOLS)}
        dataset = research_mod.build_external_regime_feature_dataset(symbol_daily, make_funding_features())

        self.assertIn("average_funding_rate", dataset.columns)
        self.assertTrue(dataset["average_funding_rate"].notna().any())

    def test_train_ext_thresholds_computed(self) -> None:
        thresholds = research_mod.compute_train_thresholds(make_feature_rows())

        self.assertEqual(thresholds["source_split"], "train_ext")
        self.assertIn("trend_efficiency_mean", thresholds["quantiles"])

    def test_validation_oos_do_not_participate_in_thresholds(self) -> None:
        features = make_feature_rows()
        baseline = research_mod.compute_train_thresholds(features)
        features.loc[features["split"] != "train_ext", "trend_efficiency_mean"] = 999.0
        changed = research_mod.compute_train_thresholds(features)

        self.assertEqual(
            baseline["quantiles"]["trend_efficiency_mean"]["q70"],
            changed["quantiles"]["trend_efficiency_mean"]["q70"],
        )
        self.assertFalse(changed["thresholds_use_validation_ext"])
        self.assertFalse(changed["thresholds_use_oos_ext"])

    def test_regime_label_generation(self) -> None:
        features = make_feature_rows()
        thresholds = research_mod.compute_train_thresholds(features)
        labels = research_mod.generate_regime_labels(features, thresholds)

        self.assertIn("regime_label", labels.columns)
        self.assertTrue(labels["is_trend_friendly"].any())
        self.assertTrue(labels["is_funding_overheated"].any())

    def test_trade_entry_time_aligns_to_completed_regime(self) -> None:
        labels = research_mod.generate_regime_labels(make_feature_rows(), research_mod.compute_train_thresholds(make_feature_rows()))
        trade = pd.DataFrame(
            [
                {
                    "split": "train_ext",
                    "policy_name": "policy_a",
                    "symbol": SYMBOLS[0],
                    "direction": "long",
                    "entry_time": pd.Timestamp("2023-01-15 00:00:00", tz="Asia/Shanghai"),
                    "exit_time": pd.Timestamp("2023-01-16 00:00:00", tz="Asia/Shanghai"),
                    "net_pnl": 1.0,
                    "no_cost_net_pnl": 1.0,
                    "funding_adjusted_net_pnl": 1.0,
                }
            ]
        )
        attribution = research_mod.align_trades_to_regime(trade, labels)

        self.assertEqual(len(attribution), 1)
        self.assertIn(attribution.iloc[0]["regime_label"], set(labels["regime_label"]))

    def test_classifier_filter_experiment(self) -> None:
        experiment = research_mod.build_classifier_filter_experiment(make_attribution())

        self.assertIn("keep_trend_friendly", set(experiment["filter_name"]))
        self.assertTrue(experiment["research_only"].all())
        self.assertTrue(experiment["not_tradable"].all())

    def test_stable_candidate_like_true_case(self) -> None:
        experiment = research_mod.build_classifier_filter_experiment(make_attribution(stable=True, concentrated=False))
        row = experiment[experiment["filter_name"] == "keep_trend_friendly"].iloc[0]

        self.assertTrue(bool(row["stable_candidate_like"]))

    def test_stable_candidate_like_false_due_to_concentration(self) -> None:
        experiment = research_mod.build_classifier_filter_experiment(make_attribution(stable=True, concentrated=True))
        row = experiment[experiment["filter_name"] == "keep_trend_friendly"].iloc[0]

        self.assertFalse(bool(row["stable_candidate_like"]))
        self.assertGreater(row["largest_symbol_pnl_share"], 0.70)

    def test_top_5pct_trade_contribution_over_0p8_is_false(self) -> None:
        experiment = research_mod.build_classifier_filter_experiment(make_top_concentrated_attribution())
        row = experiment[experiment["filter_name"] == "keep_trend_friendly"].iloc[0]

        self.assertFalse(bool(row["stable_candidate_like"]))
        self.assertGreater(row["top_5pct_trade_pnl_contribution"], 0.80)

    def test_funding_adjusted_negative_is_false(self) -> None:
        attribution = make_attribution(stable=True, concentrated=False)
        attribution.loc[attribution["split"] == "oos_ext", "funding_adjusted_net_pnl"] = -1.0
        experiment = research_mod.build_classifier_filter_experiment(attribution)
        row = experiment[experiment["filter_name"] == "keep_trend_friendly"].iloc[0]

        self.assertFalse(bool(row["stable_candidate_like"]))
        self.assertIn("oos_funding_adjusted_net_pnl_negative", row["strict_rejected_reasons"])

    def test_trade_count_under_10_is_false(self) -> None:
        attribution = make_attribution(stable=True, concentrated=False)
        attribution = attribution[~((attribution["split"] == "oos_ext") & (attribution.groupby("split").cumcount() >= 8))]
        experiment = research_mod.build_classifier_filter_experiment(attribution)
        row = experiment[experiment["filter_name"] == "keep_trend_friendly"].iloc[0]

        self.assertFalse(bool(row["stable_candidate_like"]))
        self.assertIn("oos_ext_trade_count_under_10", row["strict_rejected_reasons"])

    def test_prevents_pnl_as_feature(self) -> None:
        with self.assertRaises(research_mod.ExternalRegimeClassifierError):
            research_mod.assert_no_forbidden_feature_columns(["trend_efficiency_mean", "policy_pnl"])

    def test_output_files_exist(self) -> None:
        features = make_feature_rows()
        thresholds = research_mod.compute_train_thresholds(features)
        labels = research_mod.generate_regime_labels(features, thresholds)
        distribution = research_mod.build_regime_label_distribution(labels)
        attribution = make_attribution()
        experiment = research_mod.build_classifier_filter_experiment(attribution)
        summary = research_mod.build_research_summary(labels, distribution, attribution, experiment, thresholds)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            research_mod.write_outputs(
                output_dir,
                features,
                thresholds,
                labels,
                distribution,
                research_mod.align_trades_to_regime(attribution, labels),
                research_mod.build_policy_performance_by_regime(attribution),
                research_mod.build_split_performance_by_regime(attribution),
                experiment,
                summary,
            )
            paths = [output_dir / name for name in research_mod.REQUIRED_OUTPUT_FILES]
            summary_payload = json.loads((output_dir / "external_regime_classifier_summary.json").read_text(encoding="utf-8"))

            for path in paths:
                self.assertTrue(path.exists(), str(path))
            self.assertFalse(summary_payload["strategy_development_allowed"])

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-external-regime-classifier:", makefile)
        self.assertIn("scripts/research_external_regime_classifier.py", makefile)


if __name__ == "__main__":
    unittest.main()
