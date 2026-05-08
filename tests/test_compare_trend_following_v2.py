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

import compare_trend_following_v2 as compare_mod


def write_trend_leaderboard(directory: Path, split: str) -> None:
    """Write a minimal trend_policy_leaderboard.csv for one split."""

    directory.mkdir(parents=True, exist_ok=True)
    stable_no_cost = {"train": 4.0, "validation": 2.0, "oos": 1.0}
    stable_net = {"train": 3.5, "validation": 1.5, "oos": 0.2}
    unstable_no_cost = {"train": 4.0, "validation": -1.0, "oos": -2.0}
    unstable_net = {"train": 3.0, "validation": -1.5, "oos": -2.5}
    cost_drag_no_cost = {"train": 3.0, "validation": 2.0, "oos": 0.5}
    cost_drag_net = {"train": 2.5, "validation": 1.5, "oos": -0.1}
    validation_negative_no_cost = {"train": 3.0, "validation": -0.5, "oos": 3.0}
    validation_negative_net = {"train": 2.5, "validation": -0.8, "oos": 2.5}
    oos_big_cost_no_cost = {"train": 3.0, "validation": 2.0, "oos": 5.0}
    oos_big_cost_net = {"train": 2.5, "validation": 1.5, "oos": -10.0}
    rows = [
        {
            "policy_name": "stable_policy_atr3",
            "base_policy_name": "stable_policy",
            "atr_mult": 3.0,
            "timeframe": "1h",
            "trade_count": 12,
            "net_pnl": stable_net[split],
            "no_cost_net_pnl": stable_no_cost[split],
            "cost_drag": stable_no_cost[split] - stable_net[split],
            "max_drawdown": 1.0,
            "max_ddpercent": 5.0,
            "win_rate": 0.4,
            "profit_factor": 1.2,
            "top_5pct_trade_pnl_contribution": 0.5,
        },
        {
            "policy_name": "cost_drag_policy_atr3",
            "base_policy_name": "cost_drag_policy",
            "atr_mult": 3.0,
            "timeframe": "4h",
            "trade_count": 12,
            "net_pnl": cost_drag_net[split],
            "no_cost_net_pnl": cost_drag_no_cost[split],
            "cost_drag": cost_drag_no_cost[split] - cost_drag_net[split],
            "max_drawdown": 1.0,
            "max_ddpercent": 5.0,
            "win_rate": 0.35,
            "profit_factor": 1.1,
            "top_5pct_trade_pnl_contribution": 0.6,
        },
        {
            "policy_name": "unstable_policy_atr3",
            "base_policy_name": "unstable_policy",
            "atr_mult": 3.0,
            "timeframe": "1h",
            "trade_count": 12,
            "net_pnl": unstable_net[split],
            "no_cost_net_pnl": unstable_no_cost[split],
            "cost_drag": unstable_no_cost[split] - unstable_net[split],
            "max_drawdown": 1.0,
            "max_ddpercent": 5.0,
            "win_rate": 0.3,
            "profit_factor": 0.8,
            "top_5pct_trade_pnl_contribution": 0.9,
        },
        {
            "policy_name": "too_few_trades_atr3",
            "base_policy_name": "too_few_trades",
            "atr_mult": 3.0,
            "timeframe": "1h",
            "trade_count": 5,
            "net_pnl": 1.0,
            "no_cost_net_pnl": 2.0,
            "cost_drag": 1.0,
            "max_drawdown": 1.0,
            "max_ddpercent": 5.0,
            "win_rate": 0.4,
            "profit_factor": 1.2,
            "top_5pct_trade_pnl_contribution": 0.5,
        },
        {
            "policy_name": "validation_negative_atr3",
            "base_policy_name": "validation_negative",
            "atr_mult": 3.0,
            "timeframe": "1h",
            "trade_count": 12,
            "net_pnl": validation_negative_net[split],
            "no_cost_net_pnl": validation_negative_no_cost[split],
            "cost_drag": validation_negative_no_cost[split] - validation_negative_net[split],
            "max_drawdown": 1.0,
            "max_ddpercent": 5.0,
            "win_rate": 0.4,
            "profit_factor": 1.2,
            "top_5pct_trade_pnl_contribution": 0.5,
        },
        {
            "policy_name": "oos_big_cost_loss_atr3",
            "base_policy_name": "oos_big_cost_loss",
            "atr_mult": 3.0,
            "timeframe": "1h",
            "trade_count": 12,
            "net_pnl": oos_big_cost_net[split],
            "no_cost_net_pnl": oos_big_cost_no_cost[split],
            "cost_drag": oos_big_cost_no_cost[split] - oos_big_cost_net[split],
            "max_drawdown": 1.0,
            "max_ddpercent": 5.0,
            "win_rate": 0.4,
            "profit_factor": 1.2,
            "top_5pct_trade_pnl_contribution": 0.5,
        },
        {
            "policy_name": "high_drawdown_atr3",
            "base_policy_name": "high_drawdown",
            "atr_mult": 3.0,
            "timeframe": "4h",
            "trade_count": 12,
            "net_pnl": 2.0,
            "no_cost_net_pnl": 3.0,
            "cost_drag": 1.0,
            "max_drawdown": 40.0,
            "max_ddpercent": 35.0 if split == "validation" else 5.0,
            "win_rate": 0.4,
            "profit_factor": 1.2,
            "top_5pct_trade_pnl_contribution": 0.5,
        },
    ]
    pd.DataFrame(rows).to_csv(directory / "trend_policy_leaderboard.csv", index=False)


class CompareTrendFollowingV2Test(unittest.TestCase):
    def test_compare_identifies_stable_candidate_true_and_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_trend_leaderboard(directory, split)

            output_dir = root / "compare"
            summary = compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)
            compare_df = pd.read_csv(output_dir / "trend_compare_leaderboard.csv")

        stable = {row["policy_name"] for row in summary["stable_candidates"]}
        self.assertIn("stable_policy_atr3", stable)
        self.assertIn("cost_drag_policy_atr3", stable)
        unstable = compare_df[compare_df["policy_name"] == "unstable_policy_atr3"].iloc[0]
        too_few = compare_df[compare_df["policy_name"] == "too_few_trades_atr3"].iloc[0]
        cost_drag = compare_df[compare_df["policy_name"] == "cost_drag_policy_atr3"].iloc[0]
        validation_negative = compare_df[compare_df["policy_name"] == "validation_negative_atr3"].iloc[0]
        oos_big_cost = compare_df[compare_df["policy_name"] == "oos_big_cost_loss_atr3"].iloc[0]
        high_drawdown = compare_df[compare_df["policy_name"] == "high_drawdown_atr3"].iloc[0]

        self.assertFalse(bool(unstable["stable_candidate"]))
        self.assertFalse(bool(too_few["stable_candidate"]))
        self.assertFalse(bool(validation_negative["stable_candidate"]))
        self.assertFalse(bool(oos_big_cost["stable_candidate"]))
        self.assertFalse(bool(oos_big_cost["oos_cost_drag_explainable"]))
        self.assertFalse(bool(high_drawdown["stable_candidate"]))
        self.assertTrue(bool(cost_drag["oos_cost_drag_explainable"]))
        self.assertFalse(summary["trend_following_v2_failed"])

    def test_compare_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_trend_leaderboard(directory, split)

            output_dir = root / "compare"
            compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)

            for filename in [
                "trend_compare_summary.json",
                "trend_compare_leaderboard.csv",
                "trend_compare_report.md",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)

            payload = json.loads((output_dir / "trend_compare_summary.json").read_text(encoding="utf-8"))

        self.assertTrue(payload["stable_candidate_exists"])


if __name__ == "__main__":
    unittest.main()
