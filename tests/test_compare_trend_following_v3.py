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

import compare_trend_following_v3 as compare_mod


def write_v3_leaderboard(directory: Path, split: str) -> None:
    """Write a minimal V3 policy leaderboard for one split."""

    directory.mkdir(parents=True, exist_ok=True)
    no_cost = {
        "stable": {"train": 10.0, "validation": 8.0, "oos": 6.0},
        "oos_cost_negative": {"train": 10.0, "validation": 8.0, "oos": 6.0},
        "too_few": {"train": 10.0, "validation": 8.0, "oos": 6.0},
        "single_symbol": {"train": 10.0, "validation": 8.0, "oos": 6.0},
        "top_trade": {"train": 10.0, "validation": 8.0, "oos": 6.0},
        "validation_negative": {"train": 10.0, "validation": -1.0, "oos": 6.0},
    }
    net = {
        "stable": {"train": 9.0, "validation": 7.0, "oos": 5.0},
        "oos_cost_negative": {"train": 9.0, "validation": 7.0, "oos": -1.0},
        "too_few": {"train": 9.0, "validation": 7.0, "oos": 5.0},
        "single_symbol": {"train": 9.0, "validation": 7.0, "oos": 5.0},
        "top_trade": {"train": 9.0, "validation": 7.0, "oos": 5.0},
        "validation_negative": {"train": 9.0, "validation": -2.0, "oos": 5.0},
    }
    rows = []
    for policy in no_cost:
        trade_count = 5 if policy == "too_few" else 12
        largest_share = 0.9 if policy == "single_symbol" else 0.6
        top_share = 0.9 if policy == "top_trade" else 0.6
        rows.append(
            {
                "policy_name": policy,
                "symbol_count": 5,
                "trade_count": trade_count,
                "long_count": 6,
                "short_count": 6,
                "active_symbol_count": 1 if policy == "single_symbol" else 3,
                "no_cost_net_pnl": no_cost[policy][split],
                "net_pnl": net[policy][split],
                "fee_total": no_cost[policy][split] - net[policy][split],
                "slippage_total": 0.0,
                "cost_drag": no_cost[policy][split] - net[policy][split],
                "win_rate": 0.5,
                "profit_factor": 1.2,
                "avg_win": 2.0,
                "avg_loss": -1.0,
                "avg_trade_net_pnl": 0.5,
                "median_trade_net_pnl": 0.2,
                "max_drawdown": 1.0,
                "max_ddpercent": 5.0,
                "return_drawdown_ratio": 2.0,
                "sharpe_like": 1.0,
                "avg_holding_minutes": 240.0,
                "median_holding_minutes": 240.0,
                "top_5pct_trade_pnl_contribution": top_share,
                "best_trade": 3.0,
                "worst_trade": -1.0,
                "largest_symbol_pnl_share": largest_share,
                "portfolio_turnover": 1000.0,
                "max_concurrent_positions": 3,
            }
        )
    pd.DataFrame(rows).to_csv(directory / "trend_v3_policy_leaderboard.csv", index=False)


class CompareTrendFollowingV3Test(unittest.TestCase):
    def test_compare_stable_candidate_true_and_rejected_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_v3_leaderboard(directory, split)

            output_dir = root / "compare"
            summary = compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)
            compare_df = pd.read_csv(output_dir / "trend_v3_compare_leaderboard.csv")

        stable = {row["policy_name"] for row in summary["stable_candidates"]}
        self.assertIn("stable", stable)
        rejected = {row["policy_name"]: row["rejection_reasons"] for row in summary["rejected_candidates_with_reasons"]}
        self.assertIn("oos_cost_aware_net_pnl_negative", rejected["oos_cost_negative"])
        self.assertIn("train_trade_count_under_10", rejected["too_few"])
        self.assertIn("oos_largest_symbol_pnl_share_over_0p7", rejected["single_symbol"])
        self.assertIn("oos_top_5pct_trade_pnl_contribution_over_0p8", rejected["top_trade"])
        self.assertIn("validation_no_cost_net_pnl_not_positive", rejected["validation_negative"])
        self.assertTrue(bool(compare_df[compare_df["policy_name"] == "stable"].iloc[0]["stable_candidate"]))

    def test_compare_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            oos_dir = root / "oos"
            for split, directory in [("train", train_dir), ("validation", validation_dir), ("oos", oos_dir)]:
                write_v3_leaderboard(directory, split)

            output_dir = root / "compare"
            compare_mod.run_compare(train_dir, validation_dir, oos_dir, output_dir)

            for filename in [
                "trend_v3_compare_summary.json",
                "trend_v3_compare_leaderboard.csv",
                "trend_v3_compare_report.md",
            ]:
                self.assertTrue((output_dir / filename).exists(), filename)
            payload = json.loads((output_dir / "trend_v3_compare_summary.json").read_text(encoding="utf-8"))

        self.assertTrue(payload["stable_candidate_exists"])


if __name__ == "__main__":
    unittest.main()
