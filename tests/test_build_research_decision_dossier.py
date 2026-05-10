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


if __name__ == "__main__":
    unittest.main()
