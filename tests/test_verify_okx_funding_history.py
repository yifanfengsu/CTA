from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_okx_funding_history as download_mod
import verify_okx_funding_history as verify_mod


def write_funding_csv(path: Path, funding_times: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["inst_id", "funding_time", "funding_rate"])
        writer.writeheader()
        for timestamp in funding_times:
            writer.writerow({"inst_id": "BTC-USDT-SWAP", "funding_time": str(timestamp), "funding_rate": "0.0001"})


class VerifyOkxFundingHistoryTest(unittest.TestCase):
    def test_sorted_funding_csv_verifies_without_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            window = download_mod.parse_funding_window("2023-01-01", "2023-01-01", "UTC")
            path = download_mod.funding_csv_path(root, "BTC-USDT-SWAP", window)
            write_funding_csv(path, [1672531200000, 1672560000000, 1672588800000])

            result = verify_mod.verify_inst_id("BTC-USDT-SWAP", root, window)

            self.assertEqual(result["row_count"], 3)
            self.assertTrue(result["timestamp_strictly_increasing"])
            self.assertEqual(result["duplicate_timestamp_count"], 0)
            self.assertEqual(result["missing_or_large_gap_count"], 0)
            self.assertTrue(result["coverage_complete"])

    def test_duplicate_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            window = download_mod.parse_funding_window("2023-01-01", "2023-01-01", "UTC")
            path = download_mod.funding_csv_path(root, "BTC-USDT-SWAP", window)
            write_funding_csv(path, [1672531200000, 1672531200000, 1672560000000])

            result = verify_mod.verify_inst_id("BTC-USDT-SWAP", root, window)

            self.assertEqual(result["duplicate_timestamp_count"], 1)
            self.assertFalse(result["timestamp_strictly_increasing"])
            self.assertIn("timestamp_not_strictly_increasing", result["warnings"])

    def test_large_gap_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            window = download_mod.parse_funding_window("2023-01-01", "2023-01-03", "UTC")
            path = download_mod.funding_csv_path(root, "BTC-USDT-SWAP", window)
            write_funding_csv(path, [1672531200000, 1672560000000, 1672704000000])

            result = verify_mod.verify_inst_id("BTC-USDT-SWAP", root, window)

            self.assertGreaterEqual(result["missing_or_large_gap_count"], 1)
            self.assertTrue(result["large_gaps"])
            self.assertIn("large_gap_count=1", result["warnings"])

    def test_missing_csv_is_warning_not_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            window = download_mod.parse_funding_window("2023-01-01", "2023-01-01", "UTC")

            result = verify_mod.verify_inst_id("BTC-USDT-SWAP", Path(tmp_dir), window)

            self.assertFalse(result["exists"])
            self.assertIn("missing_funding_csv", result["warnings"])

    def test_incomplete_funding_returns_nonzero_by_default_and_zero_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            funding_dir = root / "funding"
            reports_dir = root / "reports"
            window = download_mod.parse_funding_window("2023-01-01", "2023-01-10", "UTC")
            path = download_mod.funding_csv_path(funding_dir, "BTC-USDT-SWAP", window)
            write_funding_csv(path, [1672876800000, 1672905600000])
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "okx_funding_download_summary.json").write_text(
                json.dumps(
                    {
                        "downloads": [
                            {
                                "inst_id": "BTC-USDT-SWAP",
                                "completion_status": "partial_endpoint_limited",
                                "endpoint_history_limit_suspected": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            strict_code = verify_mod.main(
                [
                    "--funding-dir",
                    str(funding_dir),
                    "--inst-ids",
                    "BTC-USDT-SWAP",
                    "--start",
                    "2023-01-01",
                    "--end",
                    "2023-01-10",
                    "--timezone",
                    "UTC",
                    "--output-dir",
                    str(reports_dir),
                ]
            )
            allow_code = verify_mod.main(
                [
                    "--funding-dir",
                    str(funding_dir),
                    "--inst-ids",
                    "BTC-USDT-SWAP",
                    "--start",
                    "2023-01-01",
                    "--end",
                    "2023-01-10",
                    "--timezone",
                    "UTC",
                    "--output-dir",
                    str(reports_dir),
                    "--allow-partial",
                ]
            )

            self.assertEqual(strict_code, 2)
            self.assertEqual(allow_code, 0)
            summary = json.loads((reports_dir / "okx_funding_verify_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["funding_data_complete"])
            self.assertIn("endpoint_lower_bound_insufficient", summary["incomplete_reason"])
            report = (reports_dir / "okx_funding_verify_report.md").read_text(encoding="utf-8")
            self.assertIn("WARNING: partial funding data accepted", report)


if __name__ == "__main__":
    unittest.main()
