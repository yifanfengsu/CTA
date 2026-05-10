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
import import_okx_funding_csv as import_mod
import verify_okx_funding_history as verify_mod


INST_ID = "BTC-USDT-SWAP"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_output_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_import_for_test(
    *,
    input_paths: list[Path],
    output_dir: Path,
    reports_dir: Path | None = None,
    start: str = "2023-01-01",
    end: str = "2023-01-02",
    append: bool = False,
) -> dict[str, object]:
    return import_mod.run_import(
        input_paths=input_paths,
        inst_id=INST_ID,
        output_dir=output_dir,
        reports_dir=reports_dir,
        start=start,
        end=end,
        timezone_name="UTC",
        append=append,
    )


class ImportOkxFundingCsvTest(unittest.TestCase):
    def test_single_file_import_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_csv = root / "manual_okx_funding.csv"
            write_csv(
                input_csv,
                ["instId", "fundingTime", "fundingRate"],
                [
                    {"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"},
                    {"instId": INST_ID, "fundingTime": "1672560000000", "fundingRate": "0.0001"},
                    {"instId": INST_ID, "fundingTime": "1672588800000", "fundingRate": "0.0001"},
                ],
            )

            output_dir = root / "funding"
            summary = run_import_for_test(
                input_paths=[input_csv],
                output_dir=output_dir,
                start="2023-01-01",
                end="2023-01-01",
            )

            self.assertEqual(summary["row_count"], 3)
            output_csv = Path(str(summary["output_csv"]))
            self.assertTrue(output_csv.exists())
            window = download_mod.parse_funding_window("2023-01-01", "2023-01-01", "UTC")
            result = verify_mod.verify_inst_id(INST_ID, output_dir, window)
            self.assertTrue(result["coverage_complete"])

    def test_multiple_file_import_merges_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "BTC-USDT-SWAP_part1.csv"
            second = root / "BTC-USDT-SWAP_part2.csv"
            write_csv(first, ["instId", "fundingTime", "fundingRate"], [{"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"}])
            write_csv(second, ["instId", "fundingTime", "fundingRate"], [{"instId": INST_ID, "fundingTime": "1672560000000", "fundingRate": "0.0002"}])

            summary = run_import_for_test(input_paths=[first, second], output_dir=root / "funding")

            self.assertEqual(summary["row_count"], 2)
            rows = read_output_rows(Path(str(summary["output_csv"])))
            self.assertEqual([row["funding_time"] for row in rows], ["1672531200000", "1672560000000"])

    def test_input_dir_discovers_matching_inst_id_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "downloads"
            write_csv(
                input_dir / "okx_BTC-USDT-SWAP_a.csv",
                ["instId", "fundingTime", "fundingRate"],
                [{"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"}],
            )
            write_csv(
                input_dir / "okx_ETH-USDT-SWAP_a.csv",
                ["instId", "fundingTime", "fundingRate"],
                [{"instId": "ETH-USDT-SWAP", "fundingTime": "1672531200000", "fundingRate": "0.0001"}],
            )

            paths = import_mod.discover_input_paths(input_path=None, inputs=None, input_dir=input_dir, inst_id=INST_ID)
            summary = run_import_for_test(input_paths=paths, output_dir=root / "funding")

            self.assertEqual([path.name for path in paths], ["okx_BTC-USDT-SWAP_a.csv"])
            self.assertEqual(summary["row_count"], 1)

    def test_duplicate_funding_time_is_deduped_last_row_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_csv = root / "dupes.csv"
            write_csv(
                input_csv,
                ["instId", "fundingTime", "fundingRate"],
                [
                    {"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"},
                    {"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0002"},
                ],
            )

            summary = run_import_for_test(input_paths=[input_csv], output_dir=root / "funding")
            rows = read_output_rows(Path(str(summary["output_csv"])))

            self.assertEqual(summary["duplicate_timestamp_count"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["funding_rate"], "0.0002")

    def test_funding_time_is_sorted_ascending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_csv = root / "unsorted.csv"
            write_csv(
                input_csv,
                ["instId", "fundingTime", "fundingRate"],
                [
                    {"instId": INST_ID, "fundingTime": "1672588800000", "fundingRate": "0.0003"},
                    {"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"},
                    {"instId": INST_ID, "fundingTime": "1672560000000", "fundingRate": "0.0002"},
                ],
            )

            summary = run_import_for_test(input_paths=[input_csv], output_dir=root / "funding")
            rows = read_output_rows(Path(str(summary["output_csv"])))

            self.assertEqual([row["funding_time"] for row in rows], ["1672531200000", "1672560000000", "1672588800000"])

    def test_start_end_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_csv = root / "window.csv"
            write_csv(
                input_csv,
                ["instId", "fundingTime", "fundingRate"],
                [
                    {"instId": INST_ID, "fundingTime": "1672444800000", "fundingRate": "0.0000"},
                    {"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"},
                    {"instId": INST_ID, "fundingTime": "1672617600000", "fundingRate": "0.0002"},
                    {"instId": INST_ID, "fundingTime": "1672790400000", "fundingRate": "0.0004"},
                ],
            )

            summary = run_import_for_test(input_paths=[input_csv], output_dir=root / "funding", start="2023-01-01", end="2023-01-02")
            rows = read_output_rows(Path(str(summary["output_csv"])))

            self.assertEqual(summary["filtered_out_of_window_count"], 2)
            self.assertEqual([row["funding_time"] for row in rows], ["1672531200000", "1672617600000"])

    def test_field_aliases_are_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_csv = root / "aliases.csv"
            write_csv(
                input_csv,
                ["instrument", "ts", "rate"],
                [{"instrument": INST_ID, "ts": "2023-01-01T00:00:00Z", "rate": "1E-4"}],
            )

            summary = run_import_for_test(input_paths=[input_csv], output_dir=root / "funding")
            rows = read_output_rows(Path(str(summary["output_csv"])))

            self.assertEqual(rows[0]["funding_time"], "1672531200000")
            self.assertEqual(rows[0]["funding_rate"], "0.0001")

    def test_missing_funding_rate_field_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_csv = Path(tmp_dir) / "missing_rate.csv"
            write_csv(input_csv, ["instId", "fundingTime"], [{"instId": INST_ID, "fundingTime": "1672531200000"}])

            with self.assertRaises(import_mod.FundingCsvImportError) as context:
                import_mod.import_rows(input_csv, INST_ID, "UTC")

            self.assertIn("missing funding rate column", str(context.exception))
            self.assertIn("fundingrate", str(context.exception))

    def test_append_mode_merges_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "first.csv"
            second = root / "second.csv"
            output_dir = root / "funding"
            write_csv(first, ["instId", "fundingTime", "fundingRate"], [{"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"}])
            write_csv(second, ["instId", "fundingTime", "fundingRate"], [{"instId": INST_ID, "fundingTime": "1672560000000", "fundingRate": "0.0002"}])

            run_import_for_test(input_paths=[first], output_dir=output_dir)
            summary = run_import_for_test(input_paths=[second], output_dir=output_dir, append=True)
            rows = read_output_rows(Path(str(summary["output_csv"])))

            self.assertEqual(summary["existing_row_count"], 1)
            self.assertEqual([row["funding_time"] for row in rows], ["1672531200000", "1672560000000"])

    def test_reports_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_csv = root / "report.csv"
            reports_dir = root / "reports"
            write_csv(input_csv, ["instId", "fundingTime", "fundingRate"], [{"instId": INST_ID, "fundingTime": "1672531200000", "fundingRate": "0.0001"}])

            summary = run_import_for_test(input_paths=[input_csv], output_dir=root / "funding", reports_dir=reports_dir)

            report_path = reports_dir / "okx_funding_import_report.md"
            summary_path = reports_dir / "okx_funding_import_summary.json"
            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["output_csv"], summary["output_csv"])


if __name__ == "__main__":
    unittest.main()
