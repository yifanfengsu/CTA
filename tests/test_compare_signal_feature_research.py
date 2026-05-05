from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_signal_feature_research as compare_mod


def write_split_outputs(directory: Path, split: str) -> None:
    """Write minimal feature research outputs for one split."""

    directory.mkdir(parents=True, exist_ok=True)
    stable_values = {"train": -0.20, "validation": -0.16, "oos": -0.11}
    unstable_values = {"train": 0.24, "validation": -0.04, "oos": 0.02}
    ic = pd.DataFrame(
        [
            {
                "feature": "breakout_distance_atr",
                "target": "future_return_60m",
                "count": 120,
                "spearman": stable_values[split],
                "abs_spearman": abs(stable_values[split]),
            },
            {
                "feature": "rsi",
                "target": "future_return_60m",
                "count": 120,
                "spearman": unstable_values[split],
                "abs_spearman": abs(unstable_values[split]),
            },
            {
                "feature": "atr_pct",
                "target": "mfe_atr",
                "count": 120,
                "spearman": 0.08,
                "abs_spearman": 0.08,
            },
        ]
    )
    bins = pd.DataFrame(
        [
            {
                "feature": "breakout_distance_atr",
                "bin": 1,
                "count": 60,
                "min": 0.1,
                "max": 0.5,
                "median_future_return_60m": 0.001,
            }
        ]
    )
    ic.to_csv(directory / "feature_ic.csv", index=False)
    bins.to_csv(directory / "feature_bins.csv", index=False)


class CompareSignalFeatureResearchTest(unittest.TestCase):
    def test_compare_identifies_consistent_and_inconsistent_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_split_outputs(directory, split)

            output_dir = root / "feature_compare"
            summary = compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)
            compare_ic = pd.read_csv(output_dir / "feature_compare_ic.csv")

        stable_features = {row["feature"] for row in summary["stable_feature_candidates"]}
        single_split_features = {row["feature"] for row in summary["single_split_only_features"]}
        self.assertIn("breakout_distance_atr", stable_features)
        self.assertIn("rsi", single_split_features)
        row = compare_ic[compare_ic["feature"] == "breakout_distance_atr"].iloc[0]
        self.assertTrue(bool(row["direction_consistent"]))

    def test_compare_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_split_outputs(directory, split)

            output_dir = root / "feature_compare"
            compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)

            for filename in [
                "feature_compare_summary.json",
                "feature_compare_ic.csv",
                "feature_compare_report.md",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)

            payload = json.loads((output_dir / "feature_compare_summary.json").read_text(encoding="utf-8"))

        self.assertFalse(payload["no_stable_feature_edge"])


if __name__ == "__main__":
    unittest.main()
