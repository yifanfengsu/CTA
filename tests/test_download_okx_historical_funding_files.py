from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_okx_historical_funding_files as hist_mod


INST_ID = "BTC-USDT-SWAP"


def csv_bytes(rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> bytes:
    output = io.StringIO()
    names = fieldnames or list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def zip_bytes(filename: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, payload)
    return buffer.getvalue()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class DownloadOkxHistoricalFundingFilesTest(unittest.TestCase):
    def test_monthly_plan_generation(self) -> None:
        periods = hist_mod.build_periods("2023-01-15", "2023-03-02", "monthly")
        self.assertEqual(
            [(period.start_arg, period.end_arg) for period in periods],
            [("2023-01-15", "2023-01-31"), ("2023-02-01", "2023-02-28"), ("2023-03-01", "2023-03-02")],
        )
        plan = hist_mod.build_download_plan(
            inst_ids=[INST_ID],
            start="2023-01-15",
            end="2023-03-02",
            aggregation="monthly",
            endpoint_url="https://www.okx.test/api/v5/public/historical-market-data",
            module="funding",
        )
        self.assertEqual(len(plan), 3)
        self.assertIn("module=3", plan[0]["request_url"])
        self.assertIn("instType=SWAP", plan[0]["request_url"])
        self.assertIn("instFamilyList=BTC-USDT", plan[0]["request_url"])
        self.assertIn("dateAggrType=monthly", plan[0]["request_url"])

    def test_csv_download_link_parsing(self) -> None:
        payload = {
            "code": "0",
            "data": [
                {"fileUrl": "https://files.okx.test/a.csv"},
                {"children": [{"downloadUrl": "https://files.okx.test/b.zip"}]},
            ],
        }
        self.assertEqual(
            hist_mod.extract_download_urls(payload),
            ["https://files.okx.test/a.csv", "https://files.okx.test/b.zip"],
        )

    def test_zip_file_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            zip_path = root / "funding.zip"
            zip_path.write_bytes(zip_bytes("nested/funding.csv", b"instId,fundingTime,fundingRate\n"))

            extracted = hist_mod.safe_extract_zip(zip_path, root / "extract")

            self.assertEqual(len(extracted), 1)
            self.assertTrue(extracted[0].exists())
            self.assertEqual(extracted[0].name, "funding.csv")

    def test_download_dry_run_does_not_download_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            def requester(_url: str) -> dict[str, object]:
                raise AssertionError("requester should not be called during dry-run")

            summary = hist_mod.run_download(
                inst_ids=[INST_ID],
                start="2023-01-01",
                end="2023-01-31",
                aggregation="monthly",
                raw_output_dir=root / "raw",
                standard_output_dir=root / "standard",
                reports_dir=root / "reports",
                endpoint_url="https://www.okx.test/api/v5/public/historical-market-data",
                module="funding",
                dry_run=True,
                requester=requester,
                downloader=lambda _url: b"",
            )

            self.assertEqual(summary["status"], "dry_run")
            self.assertFalse(any((root / "raw").glob("**/*")) if (root / "raw").exists() else False)
            self.assertTrue((root / "reports" / "okx_historical_funding_files.csv").exists())
            self.assertTrue((root / "reports" / "okx_historical_funding_download_summary.json").exists())

    def test_endpoint_unavailable_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            probe_summary = root / "probe.json"
            probe_summary.write_text(
                json.dumps(
                    {
                        "endpoint_discovery_failed": True,
                        "endpoint_available": False,
                        "funding_module_available": False,
                        "aggregation_supported": False,
                        "auth_required": False,
                        "can_auto_download": False,
                    }
                ),
                encoding="utf-8",
            )

            summary = hist_mod.run_download(
                inst_ids=[INST_ID],
                start="2023-01-01",
                end="2023-01-31",
                aggregation="monthly",
                raw_output_dir=root / "raw",
                standard_output_dir=root / "standard",
                reports_dir=root / "reports",
                probe_summary_path=probe_summary,
                dry_run=True,
            )

            self.assertEqual(summary["status"], "endpoint_unavailable")
            self.assertEqual(summary["failure_reason"], "endpoint_discovery_failed")
            report = (root / "reports" / "okx_historical_funding_download_report.md").read_text(encoding="utf-8")
            self.assertIn("manual CSV import", report)

    def test_downloads_csv_and_zip_then_imports_standard_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            url_csv = "https://files.okx.test/btc_part1.csv"
            url_zip = "https://files.okx.test/btc_part2.zip"

            def requester(url: str) -> dict[str, object]:
                self.assertIn("instFamilyList=BTC-USDT", url)
                return {"code": "0", "data": [{"fileUrl": url_csv}, {"downloadUrl": url_zip}]}

            def downloader(url: str) -> bytes:
                if url == url_csv:
                    return csv_bytes(
                        [
                            {"instrument": INST_ID, "ts": "2023-01-02T00:00:00Z", "rate": "0.0002"},
                        ],
                        ["instrument", "ts", "rate"],
                    )
                if url == url_zip:
                    return zip_bytes(
                        "funding.csv",
                        csv_bytes(
                            [
                                {"inst_id": INST_ID, "timestamp": "2023-01-01T00:00:00Z", "funding_rate": "0.0001"},
                                {"inst_id": INST_ID, "timestamp": "2023-01-02T00:00:00Z", "funding_rate": "0.0003"},
                            ],
                            ["inst_id", "timestamp", "funding_rate"],
                        ),
                    )
                raise AssertionError(f"unexpected download URL: {url}")

            summary = hist_mod.run_download(
                inst_ids=[INST_ID],
                start="2023-01-01",
                end="2023-01-02",
                aggregation="monthly",
                raw_output_dir=root / "raw",
                standard_output_dir=root / "standard",
                reports_dir=root / "reports",
                endpoint_url="https://www.okx.test/api/v5/public/historical-market-data",
                module="funding",
                timezone_name="UTC",
                dry_run=False,
                throttle_seconds=0.0,
                max_retries=1,
                overwrite=True,
                requester=requester,
                downloader=downloader,
            )

            self.assertEqual(summary["status"], "downloaded")
            self.assertEqual(summary["downloaded_file_count"], 2)
            self.assertEqual(summary["extracted_csv_count"], 2)
            output_csv = root / "standard" / "BTC-USDT-SWAP_funding_2023-01-01_2023-01-02.csv"
            self.assertTrue(output_csv.exists())
            rows = read_rows(output_csv)
            self.assertEqual([row["funding_time"] for row in rows], ["1672531200000", "1672617600000"])
            self.assertEqual(rows[1]["funding_rate"], "0.0003")
            self.assertTrue((root / "reports" / "okx_historical_funding_download_report.md").exists())

    def test_makefile_targets_exist(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
        for target in [
            "probe-funding-source:",
            "download-funding-historical-dry-run:",
            "download-funding-historical:",
        ]:
            self.assertIn(target, makefile)


if __name__ == "__main__":
    unittest.main()
