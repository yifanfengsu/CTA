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

import compare_htf_signal_research as compare_mod


def write_htf_leaderboard(directory: Path, split: str) -> None:
    """Write a minimal htf_policy_leaderboard.csv for one split."""

    directory.mkdir(parents=True, exist_ok=True)
    stable_return = {"train": 0.003, "validation": 0.002, "oos": 0.001}
    stable_expectancy = {"train": 0.25, "validation": 0.12, "oos": 0.05}
    single_return = {"train": 0.004, "validation": -0.001, "oos": -0.002}
    single_expectancy = {"train": 0.30, "validation": -0.05, "oos": -0.08}
    rows = [
        {
            "policy_name": "stable_policy",
            "signal_count": 120,
            "median_future_return_120m": stable_return[split],
            "best_expectancy_r": stable_expectancy[split],
            "positive_rate_120m": 0.55,
        },
        {
            "policy_name": "single_split_policy",
            "signal_count": 100,
            "median_future_return_120m": single_return[split],
            "best_expectancy_r": single_expectancy[split],
            "positive_rate_120m": 0.50,
        },
        {
            "policy_name": "unstable_policy",
            "signal_count": 90,
            "median_future_return_120m": -0.001 if split != "validation" else 0.002,
            "best_expectancy_r": -0.02 if split != "validation" else 0.01,
            "positive_rate_120m": 0.45,
        },
    ]
    pd.DataFrame(rows).to_csv(directory / "htf_policy_leaderboard.csv", index=False)


class CompareHtfSignalResearchTest(unittest.TestCase):
    def test_compare_identifies_stable_and_unstable_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_htf_leaderboard(directory, split)

            output_dir = root / "compare"
            summary = compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)
            compare_df = pd.read_csv(output_dir / "htf_compare_leaderboard.csv")

        stable = {row["policy_name"] for row in summary["stable_strategy_v2_candidates"]}
        single = {row["policy_name"] for row in summary["single_split_only_policies"]}
        self.assertIn("stable_policy", stable)
        self.assertIn("single_split_policy", single)
        stable_row = compare_df[compare_df["policy_name"] == "stable_policy"].iloc[0]
        self.assertTrue(bool(stable_row["stable_candidate"]))
        self.assertFalse(summary["no_stable_htf_policy"])

    def test_compare_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_htf_leaderboard(directory, split)

            output_dir = root / "compare"
            compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)

            for filename in [
                "htf_compare_summary.json",
                "htf_compare_leaderboard.csv",
                "htf_compare_report.md",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)

            payload = json.loads((output_dir / "htf_compare_summary.json").read_text(encoding="utf-8"))

        self.assertFalse(payload["no_stable_htf_policy"])


if __name__ == "__main__":
    unittest.main()
