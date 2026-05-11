from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_okx_derivatives_data_readiness as audit_mod


INST_IDS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "LINK-USDT-SWAP", "DOGE-USDT-SWAP"]
CCYS = ["BTC", "ETH", "SOL", "LINK", "DOGE"]


def write_funding_csv(path: Path, inst_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "inst_id",
                "funding_time",
                "funding_time_utc",
                "funding_time_local",
                "funding_rate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "inst_id": inst_id,
                "funding_time": "1672531200000",
                "funding_time_utc": "2023-01-01T00:00:00+00:00",
                "funding_time_local": "2023-01-01T08:00:00+08:00",
                "funding_rate": "0.0001",
            }
        )
        writer.writerow(
            {
                "inst_id": inst_id,
                "funding_time": "1774972800000",
                "funding_time_utc": "2026-03-31T16:00:00+00:00",
                "funding_time_local": "2026-04-01T00:00:00+08:00",
                "funding_rate": "-0.0001",
            }
        )


def make_endpoint_result(
    name: str,
    group: str,
    *,
    available: bool = True,
    covers: bool = True,
    segment: bool = True,
) -> dict[str, Any]:
    return {
        "endpoint_name": name,
        "endpoint_path": f"/mock/{name.lower().replace(' ', '-')}",
        "endpoint_available": available,
        "auth_required": False,
        "request_params": {"mock": "1"},
        "response_code": 200 if available else 0,
        "response_ok": available,
        "row_count": 1 if available else 0,
        "sample_fields": "ts,value",
        "first_timestamp": "2023-01-01T00:00:00+00:00" if covers else "2026-03-01T00:00:00+00:00",
        "last_timestamp": "2026-03-31T00:00:00+00:00" if covers else "2026-03-01T00:00:00+00:00",
        "period": "1D",
        "max_window_limit": "mock",
        "can_segment_download": segment and available,
        "can_cover_2023_2026": covers,
        "usable_for_research": covers,
        "warning": "",
        "next_step": "segment download" if covers else "historical files",
        "failure_reason": "" if available else "network_failed",
        "feature_group": group,
    }


class AuditOkxDerivativesDataReadinessTest(unittest.TestCase):
    def test_endpoint_probe_result_parser(self) -> None:
        parsed = audit_mod.parse_probe_payload(
            {
                "code": "0",
                "msg": "",
                "data": [
                    {"ts": "1772323200000", "buyVol": "1", "sellVol": "2"},
                    {"ts": "1772409600000", "buyVol": "3", "sellVol": "4"},
                ],
            }
        )

        self.assertEqual(parsed["okx_code"], "0")
        self.assertEqual(parsed["row_count"], 2)
        self.assertIn("buyVol", parsed["sample_fields"])
        self.assertEqual(parsed["first_timestamp"], "2026-03-01T00:00:00+00:00")
        self.assertEqual(parsed["last_timestamp"], "2026-03-02T00:00:00+00:00")

    def test_endpoint_failure_does_not_crash(self) -> None:
        spec = audit_mod.endpoint_specs()[0]

        def failing_requester(_path: str, _params: dict[str, str], _timeout: int) -> audit_mod.HttpProbe:
            raise RuntimeError("boom")

        result = audit_mod.probe_endpoint(
            spec,
            inst_id="BTC-USDT-SWAP",
            ccy="BTC",
            start="2023-01-01",
            end="2026-03-31",
            timezone_name="Asia/Shanghai",
            probe_date="2026-03-01",
            local_funding_complete=False,
            requester=failing_requester,
        )

        self.assertFalse(result["endpoint_available"])
        self.assertFalse(result["response_ok"])
        self.assertIn("RuntimeError", result["failure_reason"])

    def test_feature_tier_classification(self) -> None:
        endpoint_results = [
            make_endpoint_result("Funding Rate History", "funding"),
            make_endpoint_result("Mark Price", "mark_index_basis", covers=False, segment=False),
            make_endpoint_result("Index Ticker", "mark_index_basis", covers=False, segment=False),
            make_endpoint_result("Taker Buy/Sell Volume", "taker_flow"),
            make_endpoint_result("Contract Long/Short Account Ratio", "long_short_ratio"),
            make_endpoint_result("Contracts Open Interest and Volume", "open_interest"),
            make_endpoint_result("Contract Open Interest History", "open_interest"),
            make_endpoint_result("Mark Price Candles History", "mark_index_basis"),
            make_endpoint_result("Index Candles History", "mark_index_basis"),
            make_endpoint_result("Premium History", "mark_index_basis", available=False, covers=False),
        ]
        rows = audit_mod.build_feature_rows(
            endpoint_results,
            start="2023-01-01",
            end="2026-03-31",
            local_funding_complete=True,
        )
        tiers = {row["feature_name"]: row["tier"] for row in rows}

        self.assertEqual(tiers["actual funding rate"], "Tier 1")
        self.assertEqual(tiers["taker buy/sell volume"], "Tier 2")
        self.assertEqual(tiers["premium/basis historical"], "Tier 3")

    def test_can_cover_2023_2026_judgement(self) -> None:
        spec = next(item for item in audit_mod.endpoint_specs() if item.name == "Taker Buy/Sell Volume")

        self.assertTrue(
            audit_mod.decide_can_cover(
                spec,
                recent_ok=True,
                coverage_ok=True,
                coverage_row_count=1,
                local_funding_complete=False,
            )
        )
        self.assertFalse(
            audit_mod.decide_can_cover(
                spec,
                recent_ok=True,
                coverage_ok=True,
                coverage_row_count=0,
                local_funding_complete=False,
            )
        )

    def test_decision_rule_true_case(self) -> None:
        endpoint_results = [
            make_endpoint_result("Contracts Open Interest and Volume", "open_interest"),
            make_endpoint_result("Taker Buy/Sell Volume", "taker_flow"),
            make_endpoint_result("Contract Long/Short Account Ratio", "long_short_ratio"),
            make_endpoint_result("Premium History", "mark_index_basis", available=False, covers=False),
        ]
        feature_rows = audit_mod.build_feature_rows(
            endpoint_results,
            start="2023-01-01",
            end="2026-03-31",
            local_funding_complete=True,
        )

        decision = audit_mod.evaluate_decision(
            feature_rows,
            endpoint_results,
            local_funding_complete=True,
        )

        self.assertTrue(decision["can_enter_derivatives_confirmed_trend_research"])
        self.assertFalse(decision["strategy_development_allowed"])
        self.assertFalse(decision["demo_live_allowed"])
        self.assertEqual(decision["recommended_next_step"], "download derivatives metrics")

    def test_decision_rule_false_case(self) -> None:
        endpoint_results = [
            make_endpoint_result("Contracts Open Interest and Volume", "open_interest", covers=False),
            make_endpoint_result("Taker Buy/Sell Volume", "taker_flow", covers=False),
        ]
        feature_rows = audit_mod.build_feature_rows(
            endpoint_results,
            start="2023-01-01",
            end="2026-03-31",
            local_funding_complete=True,
        )

        decision = audit_mod.evaluate_decision(
            feature_rows,
            endpoint_results,
            local_funding_complete=True,
        )

        self.assertFalse(decision["can_enter_derivatives_confirmed_trend_research"])
        self.assertEqual(decision["recommended_next_step"], "pause research")
        self.assertIn("open_interest_feature_missing_or_not_2023_2026", decision["blocking_reasons"])

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            funding_dir = root / "data" / "funding" / "okx"
            original_default = audit_mod.DEFAULT_FUNDING_DIR
            try:
                audit_mod.DEFAULT_FUNDING_DIR = funding_dir
                for inst_id in INST_IDS:
                    write_funding_csv(funding_dir / f"{inst_id}_funding_2023-01-01_2026-03-31.csv", inst_id)

                output_dir = root / "reports" / "research" / "derivatives_data_readiness"
                payload = audit_mod.run_audit(
                    inst_ids=INST_IDS,
                    ccys=CCYS,
                    start="2023-01-01",
                    end="2026-03-31",
                    timezone_name="Asia/Shanghai",
                    output_dir=output_dir,
                    probe_date="2026-03-01",
                    dry_run=True,
                )
            finally:
                audit_mod.DEFAULT_FUNDING_DIR = original_default

            expected = [
                output_dir / "derivatives_data_readiness_report.md",
                output_dir / "derivatives_data_readiness.json",
                output_dir / "endpoint_probe_results.csv",
                output_dir / "proposed_derivatives_features.csv",
                output_dir / "derivatives_data_download_plan.csv",
                output_dir / "unavailable_derivatives_features.csv",
            ]
            for path in expected:
                self.assertTrue(path.exists(), str(path))
            data = json.loads((output_dir / "derivatives_data_readiness.json").read_text(encoding="utf-8"))
            self.assertFalse(data["decision"]["strategy_development_allowed"])
            self.assertIn("output_paths", payload)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("audit-derivatives-data:", makefile)
        self.assertIn("scripts/audit_okx_derivatives_data_readiness.py", makefile)


if __name__ == "__main__":
    unittest.main()
