from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_alpha_sweep as alpha_sweep


class AlphaSweepConfigTest(unittest.TestCase):
    def test_build_candidate_runs_is_bounded_shortlist(self) -> None:
        base_setting = {
            "fixed_size": 0.02,
            "risk_per_trade": 0.001,
            "max_leverage": 1.0,
            "max_notional_ratio": 1.0,
            "max_trades_per_day": 20,
        }

        candidates = alpha_sweep.build_candidate_runs(
            base_setting=base_setting,
            output_dir=PROJECT_ROOT / "reports" / "tmp_alpha_sweep_test",
            max_runs=30,
        )

        self.assertLessEqual(len(candidates), 10)
        self.assertEqual(len(candidates), len(alpha_sweep.SHORTLIST_CANDIDATES))
        self.assertEqual(candidates[0].name, "baseline")
        self.assertEqual(candidates[-1].name, "conservative_combo")

    def test_guardrails_do_not_allow_expanded_risk(self) -> None:
        guarded = alpha_sweep.enforce_guardrails(
            {
                "fixed_size": 0.5,
                "risk_per_trade": 0.003,
                "max_leverage": 2.0,
                "max_notional_ratio": 1.5,
                "max_trades_per_day": 50,
            }
        )

        self.assertEqual(guarded["fixed_size"], 0.01)
        self.assertLessEqual(guarded["risk_per_trade"], 0.0005)
        self.assertLessEqual(guarded["max_leverage"], 0.5)
        self.assertLessEqual(guarded["max_notional_ratio"], 0.5)
        self.assertLessEqual(guarded["max_trades_per_day"], 10)


if __name__ == "__main__":
    unittest.main()
