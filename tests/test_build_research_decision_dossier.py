from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_research_decision_dossier as dossier_mod


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_gate_audit_fixture(root: Path) -> None:
    gate_dir = root / "reports/research/external_regime_classifier_gate_audit"
    write_text(
        gate_dir / "external_regime_gate_audit_report.md",
        "# External Regime Classifier Gate Audit\n\nNo strict stable candidate.\n",
    )
    write_json(
        gate_dir / "external_regime_gate_audit_summary.json",
        {
            "original_all_strict_stable_candidate_like_count": 0,
            "strict_stable_candidate_like_count": 0,
            "can_enter_research_only_v3_1_classifier_experiment": False,
            "strategy_development_allowed": False,
            "demo_live_allowed": False,
            "original_all_does_not_bypass_strict_gate": True,
            "top_concentration_gate_enforced": True,
            "symbol_concentration_gate_enforced": True,
            "filter_did_not_affect_oos_count": 27,
            "filter_did_affect_oos_count": 33,
            "v3_1d_ema_exclude_filters_did_not_affect_oos": True,
            "reason": "No non-original classifier filter passed strict gates after Dossier-consistent concentration checks.",
        },
    )
    write_text(
        gate_dir / "gate_comparison.csv",
        "\n".join(
            [
                "filter_name,policy_name,oos_top_5pct_trade_pnl_contribution,old_stable_candidate_like,strict_stable_candidate_like,rejected_reasons",
                "original_all,v3_1d_ema_50_200_atr5,1.9817725334062826,False,False,oos_top_5pct_trade_pnl_contribution_over_0p8",
            ]
        ),
    )
    write_text(
        gate_dir / "filter_trade_set_diff.csv",
        "\n".join(
            [
                "filter_name,policy_name,split,original_trade_count,filtered_trade_count,removed_trade_count,did_filter_change_trade_set",
                "exclude_hostile_chop_overheated,v3_1d_ema_50_200_atr5,oos_ext,36,36,0,False",
                "exclude_funding_overheated,v3_1d_ema_50_200_atr5,oos_ext,36,36,0,False",
                "keep_trend_friendly,v3_1d_ema_50_200_atr5,oos_ext,36,0,36,True",
            ]
        ),
    )


class BuildResearchDecisionDossierTest(unittest.TestCase):
    def run_temp_dossier(self) -> tuple[Path, dict[str, object]]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name) / "project"
        output_dir = Path(tmpdir.name) / "dossier"
        root.mkdir()
        payload = dossier_mod.build_dossier(
            output_dir=output_dir,
            include_existing_reports=True,
            project_root=root,
            logger=logging.getLogger("test_build_research_decision_dossier"),
        )
        return output_dir, payload

    def run_temp_dossier_with_actual_funding(self) -> tuple[Path, dict[str, object]]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name) / "project"
        output_dir = Path(tmpdir.name) / "dossier"
        root.mkdir()
        write_text(
            root / "reports/research/trend_following_v3_actual_funding/actual_funding_report.md",
            "# Actual Funding\nfunding_data_complete=true\n",
        )
        write_json(
            root / "reports/research/trend_following_v3_actual_funding/actual_funding_summary.json",
            {
                "funding_data_complete": True,
                "funding_adjusted_stable_candidate_exists": False,
                "funding_adjusted_stable_candidates": [],
                "can_enter_funding_aware_v3_1_research": False,
                "strategy_development_allowed": False,
                "demo_live_allowed": False,
                "target_policy": "v3_1d_ema_50_200_atr5",
            },
        )
        write_text(root / "reports/research/funding/okx_funding_verify_report.md", "# Verify\ncomplete\n")
        write_json(
            root / "reports/research/funding/okx_funding_verify_summary.json",
            {
                "funding_data_complete": True,
                "results": [
                    {
                        "inst_id": "BTC-USDT-SWAP",
                        "row_count": 3558,
                        "first_available_time": "2022-12-31T16:00:00+00:00",
                        "last_available_time": "2026-03-31T08:00:00+00:00",
                        "coverage_complete": True,
                        "completion_status": "complete",
                    }
                ],
                "symbols_with_warnings": [],
                "incomplete_reason": [],
            },
        )
        write_text(root / "reports/research/funding_historical_download/okx_historical_funding_download_report.md", "# Download\n")
        write_json(
            root / "reports/research/funding_historical_download/okx_historical_funding_download_summary.json",
            {
                "status": "downloaded",
                "downloaded_file_count": 195,
                "extracted_csv_count": 195,
                "inst_results": [
                    {
                        "inst_id": "BTC-USDT-SWAP",
                        "row_count": 3558,
                        "first_time": "2022-12-31T16:00:00+00:00",
                        "last_time": "2026-03-31T08:00:00+00:00",
                    }
                ],
            },
        )
        payload = dossier_mod.build_dossier(
            output_dir=output_dir,
            include_existing_reports=True,
            project_root=root,
            logger=logging.getLogger("test_build_research_decision_dossier"),
        )
        return output_dir, payload

    def run_temp_dossier_with_gate_audit(self) -> tuple[Path, dict[str, object]]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name) / "project"
        output_dir = Path(tmpdir.name) / "dossier"
        root.mkdir()
        write_gate_audit_fixture(root)
        payload = dossier_mod.build_dossier(
            output_dir=output_dir,
            include_existing_reports=True,
            project_root=root,
            logger=logging.getLogger("test_build_research_decision_dossier"),
        )
        return output_dir, payload

    def test_missing_reports_warn_without_crashing(self) -> None:
        output_dir, payload = self.run_temp_dossier()

        self.assertTrue(output_dir.exists())
        self.assertTrue(payload["warnings"])
        self.assertTrue(any(str(warning).startswith("missing_report:") for warning in payload["warnings"]))

    def test_generates_research_decision_dossier_md(self) -> None:
        output_dir, _payload = self.run_temp_dossier()

        self.assertTrue((output_dir / "research_decision_dossier.md").exists())

    def test_generates_research_decision_dossier_json(self) -> None:
        output_dir, _payload = self.run_temp_dossier()

        self.assertTrue((output_dir / "research_decision_dossier.json").exists())
        loaded = json.loads((output_dir / "research_decision_dossier.json").read_text(encoding="utf-8"))
        self.assertIn("failed_policy_families", loaded)

    def test_failed_policy_families_csv_exists(self) -> None:
        output_dir, _payload = self.run_temp_dossier()

        self.assertTrue((output_dir / "failed_policy_families.csv").exists())

    def test_do_not_continue_list_csv_exists(self) -> None:
        output_dir, _payload = self.run_temp_dossier()

        self.assertTrue((output_dir / "do_not_continue_list.csv").exists())

    def test_json_strategy_development_allowed_false(self) -> None:
        _output_dir, payload = self.run_temp_dossier()

        self.assertFalse(payload["strategy_development_allowed"])

    def test_json_demo_live_allowed_false(self) -> None:
        _output_dir, payload = self.run_temp_dossier()

        self.assertFalse(payload["demo_live_allowed"])

    def test_json_proceed_to_v3_1_research_false(self) -> None:
        _output_dir, payload = self.run_temp_dossier()

        self.assertFalse(payload["proceed_to_v3_1_research"])

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-dossier:", makefile)

    def test_actual_funding_complete_sets_json_field(self) -> None:
        _output_dir, payload = self.run_temp_dossier_with_actual_funding()

        self.assertTrue(payload["actual_funding_data_complete"])
        self.assertEqual(payload["actual_funding_source"], "OKX Historical Market Data")
        self.assertTrue(payload["current_universe_funding_complete"])

    def test_funding_adjusted_no_stable_candidate_keeps_strategy_blocked(self) -> None:
        _output_dir, payload = self.run_temp_dossier_with_actual_funding()

        self.assertFalse(payload["funding_adjusted_stable_candidate_exists"])
        self.assertFalse(payload["strategy_development_allowed"])

    def test_actual_funding_dossier_keeps_demo_live_blocked(self) -> None:
        _output_dir, payload = self.run_temp_dossier_with_actual_funding()

        self.assertFalse(payload["demo_live_allowed"])
        self.assertFalse(payload["can_enter_funding_aware_v3_1_research"])

    def test_classifier_gate_audit_complete_fields_written_to_json(self) -> None:
        output_dir, payload = self.run_temp_dossier_with_gate_audit()

        self.assertTrue(payload["external_regime_classifier_gate_audit_complete"])
        self.assertTrue(payload["classifier_old_gate_inconsistent"])
        self.assertFalse(payload["classifier_strict_stable_candidate_exists"])
        self.assertFalse(payload["can_enter_research_only_v3_1_classifier_experiment"])
        loaded = json.loads((output_dir / "research_decision_dossier.json").read_text(encoding="utf-8"))
        self.assertTrue(loaded["external_regime_classifier_gate_audit_complete"])
        self.assertFalse(loaded["classifier_strict_stable_candidate_exists"])

    def test_final_archive_fields_stay_closed_after_classifier_gate_audit(self) -> None:
        _output_dir, payload = self.run_temp_dossier_with_gate_audit()

        self.assertTrue(payload["final_current_trend_family_archived"])
        self.assertFalse(payload["final_strategy_development_allowed"])
        self.assertFalse(payload["final_demo_live_allowed"])
        self.assertFalse(payload["demo_live_allowed"])
        self.assertFalse(payload["external_classifier_rescued_v3_family"])

    def test_do_not_continue_contains_v3_1d_ema_policy(self) -> None:
        _output_dir, payload = self.run_temp_dossier_with_actual_funding()

        items = [str(row["item"]) for row in payload["do_not_continue"]]
        self.assertTrue(any("v3_1d_ema_50_200_atr5" in item for item in items))

    def test_do_not_continue_contains_external_classifier_rescue_stop(self) -> None:
        _output_dir, payload = self.run_temp_dossier_with_gate_audit()

        items = [str(row["item"]) for row in payload["do_not_continue"]]
        self.assertTrue(any("external regime classifier as V3.1 rescue" in item for item in items))

    def test_missing_actual_funding_report_warns_without_crashing(self) -> None:
        output_dir, payload = self.run_temp_dossier()

        self.assertTrue(output_dir.exists())
        self.assertTrue(any("actual_funding_report" in str(warning) for warning in payload["warnings"]))

    def test_missing_gate_audit_summary_warns_without_crashing(self) -> None:
        output_dir, payload = self.run_temp_dossier()

        self.assertTrue(output_dir.exists())
        self.assertFalse(payload["external_regime_classifier_gate_audit_complete"])
        self.assertFalse(payload["classifier_strict_stable_candidate_exists"])
        self.assertFalse(payload["can_enter_research_only_v3_1_classifier_experiment"])
        self.assertTrue(any("missing_external_regime_classifier_gate_audit_summary" in str(warning) for warning in payload["warnings"]))


if __name__ == "__main__":
    unittest.main()
