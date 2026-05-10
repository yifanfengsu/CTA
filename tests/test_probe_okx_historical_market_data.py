from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import probe_okx_historical_market_data as probe_mod


class ProbeOkxHistoricalMarketDataTest(unittest.TestCase):
    def test_probe_parses_endpoint_available_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "probe"

            def requester(url: str) -> dict[str, object]:
                self.assertIn("module=3", url)
                self.assertIn("dateAggrType=monthly", url)
                self.assertIn("instFamilyList=BTC-USDT", url)
                self.assertIn("begin=1772294400000", url)
                self.assertIn("end=1774886400000", url)
                return {
                    "code": "0",
                    "data": [
                        {
                            "instId": "BTC-USDT-SWAP",
                            "fileUrl": "https://static.okx.test/BTC-USDT-SWAP_funding_2023-01.csv",
                        }
                    ],
                }

            summary = probe_mod.run_probe(
                inst_ids=["BTC-USDT-SWAP"],
                start="2023-01-01",
                end="2026-03-31",
                data_type="funding",
                aggregation="monthly",
                output_dir=output_dir,
                dry_run=True,
                endpoint_url="https://www.okx.test/api/v5/public/historical-market-data",
                module="funding",
                requester=requester,
            )

            self.assertTrue(summary["endpoint_available"])
            self.assertTrue(summary["funding_module_available"])
            self.assertTrue(summary["aggregation_supported"])
            self.assertFalse(summary["auth_required"])
            self.assertTrue(summary["can_auto_download"])
            self.assertEqual(summary["response_kind"], "download_link")
            self.assertTrue((output_dir / "okx_historical_market_data_probe.json").exists())
            payload = json.loads((output_dir / "okx_historical_market_data_probe.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["can_auto_download"])

    def test_endpoint_discovery_failure_is_reported_without_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "probe"

            summary = probe_mod.run_probe(
                inst_ids=["BTC-USDT-SWAP"],
                start="2023-01-01",
                end="2026-03-31",
                data_type="funding",
                aggregation="monthly",
                output_dir=output_dir,
                dry_run=True,
                docs_fetcher=lambda _url: "Get historical market data is in changelog, but this page has no GET path.",
                requester=lambda _url: {"code": "0"},
            )

            self.assertTrue(summary["endpoint_discovery_failed"])
            self.assertFalse(summary["endpoint_available"])
            self.assertFalse(summary["can_auto_download"])
            self.assertEqual(summary["probe_error"], "endpoint path could not be confirmed from official docs")
            report = (output_dir / "okx_historical_market_data_probe_report.md").read_text(encoding="utf-8")
            self.assertIn("endpoint_discovery_failed=true", report)


if __name__ == "__main__":
    unittest.main()
