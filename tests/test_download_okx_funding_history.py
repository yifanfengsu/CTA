from __future__ import annotations

import json
import logging
from urllib.parse import parse_qs, urlparse
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_okx_funding_history as funding_mod


def okx_payload(inst_id: str, timestamps: list[int]) -> dict[str, object]:
    return {
        "code": "0",
        "msg": "",
        "data": [
            {
                "instId": inst_id,
                "fundingTime": str(timestamp),
                "fundingRate": "0.0001",
                "realizedRate": "0.0001",
            }
            for timestamp in timestamps
        ],
    }


class DownloadOkxFundingHistoryTest(unittest.TestCase):
    def test_parse_okx_funding_response_uses_realized_rate(self) -> None:
        payload = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingTime": "1704067200000",
                    "fundingRate": "0.0002",
                    "realizedRate": "0.00019",
                    "method": "next_period",
                    "formulaType": "withRate",
                }
            ],
        }

        rows = funding_mod.parse_okx_funding_response(payload, "Asia/Shanghai")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["inst_id"], "BTC-USDT-SWAP")
        self.assertEqual(rows[0]["funding_rate"], "0.00019")
        self.assertEqual(rows[0]["raw_funding_rate"], "0.0002")
        self.assertIn("2024-01-01T00:00:00+00:00", rows[0]["funding_time_utc"])

    def test_build_download_plan_uses_backward_after_cursor(self) -> None:
        window = funding_mod.parse_funding_window("2023-01-01", "2023-01-02", "Asia/Shanghai")

        plan = funding_mod.build_download_plan(["BTC-USDT-SWAP"], window, Path("data/funding/okx"))

        self.assertEqual(plan[0]["inst_id"], "BTC-USDT-SWAP")
        self.assertEqual(plan[0]["pagination"], "backward_by_after_cursor_using_oldest_fundingTime")
        self.assertIn("after=", plan[0]["endpoint"])
        self.assertIn("limit=400", plan[0]["endpoint"])

    def test_dry_run_writes_reports_but_no_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "funding"
            reports_dir = root / "reports"
            window = funding_mod.parse_funding_window("2023-01-01", "2023-01-02", "Asia/Shanghai")

            summary = funding_mod.run_download(
                inst_ids=["BTC-USDT-SWAP"],
                window=window,
                output_dir=output_dir,
                reports_dir=reports_dir,
                dry_run=True,
                throttle_seconds=0.0,
                max_retries=1,
                logger=logging.getLogger("test_download_funding"),
            )

            self.assertTrue(summary["success"])
            self.assertFalse(list(output_dir.glob("*.csv")))
            self.assertTrue((reports_dir / "okx_funding_download_report.md").exists())
            payload = json.loads((reports_dir / "okx_funding_download_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])

    def test_backward_pagination_continues_after_short_page_and_marks_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "funding"
            reports_dir = root / "reports"
            inst_id = "BTC-USDT-SWAP"
            window = funding_mod.parse_funding_window("2023-01-01", "2023-01-10", "UTC")
            first_page_times = [1672876800000, 1672905600000]
            calls: list[str] = []

            def requester(url: str) -> dict[str, object]:
                calls.append(url)
                query = parse_qs(urlparse(url).query)
                after = int(query["after"][0])
                if after == window.end_exclusive_ms:
                    return okx_payload(inst_id, first_page_times)
                if after == min(first_page_times):
                    return okx_payload(inst_id, [])
                self.fail(f"unexpected after cursor: {after}")

            summary = funding_mod.run_download(
                inst_ids=[inst_id],
                window=window,
                output_dir=output_dir,
                reports_dir=reports_dir,
                dry_run=False,
                throttle_seconds=0.0,
                max_retries=1,
                logger=logging.getLogger("test_download_funding"),
                requester=requester,
            )

            self.assertEqual(len(calls), 2)
            self.assertIn(f"after={min(first_page_times)}", calls[1])
            self.assertFalse(summary["funding_data_complete"])
            self.assertTrue(summary["endpoint_history_limit_suspected"])
            self.assertEqual(summary["downloads"][0]["completion_status"], "partial_endpoint_limited")
            self.assertEqual(summary["downloads"][0]["request_count"], 2)
            self.assertTrue((reports_dir / "okx_funding_download_requests.csv").exists())


if __name__ == "__main__":
    unittest.main()
