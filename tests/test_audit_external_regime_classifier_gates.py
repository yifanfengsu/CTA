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

import audit_external_regime_classifier_gates as audit_mod
import research_external_regime_classifier as research_mod
from tests.test_research_external_regime_classifier import make_attribution, make_top_concentrated_attribution


class AuditExternalRegimeClassifierGatesTest(unittest.TestCase):
    def write_classifier_fixture(self, root: Path, attribution: pd.DataFrame, *, old_stable_true: bool = False) -> Path:
        classifier_dir = root / "reports" / "research" / "external_regime_classifier"
        classifier_dir.mkdir(parents=True, exist_ok=True)
        attribution.to_csv(classifier_dir / "trade_regime_classifier_attribution.csv", index=False)
        experiment = research_mod.build_classifier_filter_experiment(attribution)
        if old_stable_true:
            mask = (experiment["filter_name"] == "original_all") & (experiment["policy_name"] == "policy_a")
            experiment.loc[mask, "stable_candidate_like"] = True
        experiment.to_csv(classifier_dir / "classifier_filter_experiment.csv", index=False)
        return classifier_dir

    def test_original_all_does_not_bypass_strict_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classifier_dir = self.write_classifier_fixture(root, make_top_concentrated_attribution(), old_stable_true=True)
            summary = audit_mod.run_audit(classifier_dir, root / "gate_audit")
            gate = pd.read_csv(root / "gate_audit" / "gate_comparison.csv")
            original = gate[(gate["filter_name"] == "original_all") & (gate["policy_name"] == "policy_a")].iloc[0]

        self.assertEqual(str(original["old_stable_candidate_like"]), "True")
        self.assertEqual(str(original["strict_stable_candidate_like"]), "False")
        self.assertIn("oos_top_5pct_trade_pnl_contribution_over_0p8", original["rejected_reasons"])
        self.assertTrue(summary["original_all_does_not_bypass_strict_gate"])

    def test_filter_trade_set_diff_detects_unchanged_oos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classifier_dir = self.write_classifier_fixture(root, make_attribution(stable=True, concentrated=False))
            audit_mod.run_audit(classifier_dir, root / "gate_audit")
            diff = pd.read_csv(root / "gate_audit" / "filter_trade_set_diff.csv")
            row = diff[
                (diff["filter_name"] == "exclude_funding_overheated")
                & (diff["policy_name"] == "policy_a")
                & (diff["split"] == "oos_ext")
            ].iloc[0]

        self.assertEqual(row["original_trade_count"], row["filtered_trade_count"])
        self.assertEqual(str(row["did_filter_change_trade_set"]), "False")

    def test_gate_comparison_and_report_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classifier_dir = self.write_classifier_fixture(root, make_top_concentrated_attribution(), old_stable_true=True)
            gate_audit_dir = root / "gate_audit"
            summary = audit_mod.run_audit(classifier_dir, gate_audit_dir)
            expected = [gate_audit_dir / name for name in audit_mod.REQUIRED_OUTPUT_FILES]
            summary_payload = json.loads((gate_audit_dir / "external_regime_gate_audit_summary.json").read_text(encoding="utf-8"))

            for path in expected:
                self.assertTrue(path.exists(), str(path))
            self.assertFalse(summary_payload["strategy_development_allowed"])
            self.assertIn("can_enter_research_only_v3_1_classifier_experiment", summary)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("audit-external-regime-gates:", makefile)
        self.assertIn("scripts/audit_external_regime_classifier_gates.py", makefile)


if __name__ == "__main__":
    unittest.main()
