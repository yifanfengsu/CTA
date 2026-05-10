from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_external_regime_classifier_feasibility as audit_mod
from history_time_utils import parse_history_range


SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
    "LINKUSDT_SWAP_OKX.GLOBAL",
    "DOGEUSDT_SWAP_OKX.GLOBAL",
]


def write_history_db(path: Path, symbols: list[str], *, minutes: int = 1440) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            (
                "create table dbbardata("
                "symbol text, exchange text, interval text, datetime text, open_interest real)"
            )
        )
        rows = []
        for vt_symbol in symbols:
            symbol, exchange = vt_symbol.split(".", maxsplit=1)
            current = datetime.fromisoformat("2023-01-01 00:00:00")
            for _ in range(minutes):
                rows.append((symbol, exchange, "1m", current.isoformat(sep=" ", timespec="seconds"), 0.0))
                current += timedelta(minutes=1)
        connection.executemany(
            "insert into dbbardata(symbol, exchange, interval, datetime, open_interest) values (?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def write_funding_csv(path: Path, inst_id: str, *, rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = datetime.fromisoformat("2023-01-01 00:00:00")
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
        for index in range(rows):
            writer.writerow(
                {
                    "inst_id": inst_id,
                    "funding_time": "",
                    "funding_time_utc": "",
                    "funding_time_local": current.isoformat(timespec="seconds"),
                    "funding_rate": "0.0001" if index % 2 == 0 else "-0.0001",
                }
            )
            current += timedelta(hours=8)


def write_all_funding(root: Path, symbols: list[str], *, start: str, end: str, rows: int = 3) -> Path:
    funding_dir = root / "data" / "funding" / "okx"
    for vt_symbol in symbols:
        inst_id = audit_mod.vt_symbol_to_okx_inst_id(vt_symbol)
        write_funding_csv(funding_dir / f"{inst_id}_funding_{start}_{end}.csv", inst_id, rows=rows)
    return funding_dir


class ExternalRegimeClassifierFeasibilityAuditTest(unittest.TestCase):
    def make_range(self, start: str = "2023-01-01", end: str = "2023-01-01") -> Any:
        return parse_history_range(
            start_arg=start,
            end_arg=end,
            interval_delta=timedelta(minutes=1),
            timezone_name="Asia/Shanghai",
        )

    def run_temp_audit(
        self,
        root: Path,
        *,
        symbols: list[str] | None = None,
        start: str = "2023-01-01",
        end: str = "2023-01-01",
        minutes: int = 1440,
        funding_rows: int = 3,
    ) -> dict[str, Any]:
        use_symbols = symbols or SYMBOLS
        db_path = root / "database.db"
        write_history_db(db_path, use_symbols, minutes=minutes)
        funding_dir = write_all_funding(root, use_symbols, start=start, end=end, rows=funding_rows)
        return audit_mod.run_audit(
            symbols=use_symbols,
            start=start,
            end=end,
            timezone_name="Asia/Shanghai",
            output_dir=root / "reports" / "research" / "external_regime_feasibility",
            database_path=db_path,
            funding_dir=funding_dir,
        )

    def test_identifies_market_data_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = self.run_temp_audit(Path(tmp_dir))

        self.assertTrue(payload["data_status"]["market_data"]["market_data_complete"])
        self.assertEqual(len(payload["data_status"]["market_data"]["symbols"]), 5)

    def test_identifies_funding_data_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = self.run_temp_audit(Path(tmp_dir))

        self.assertTrue(payload["data_status"]["funding_data"]["funding_data_complete"])
        self.assertEqual(payload["data_status"]["funding_data"]["funding_source"], audit_mod.FUNDING_SOURCE)

    def test_lists_internal_market_features(self) -> None:
        rows = audit_mod.build_feature_rows(
            market_data_complete=True,
            funding_data_complete=True,
            coverage_start="2023-01-01",
            coverage_end="2026-03-31",
        )
        names = {row["feature_name"] for row in rows if row["feature_group"] == "internal_market_features"}

        self.assertIn("trend breadth", names)
        self.assertIn("cross-symbol correlation", names)
        self.assertIn("number of symbols above EMA50/EMA200", names)

    def test_lists_funding_features(self) -> None:
        rows = audit_mod.build_feature_rows(
            market_data_complete=True,
            funding_data_complete=True,
            coverage_start="2023-01-01",
            coverage_end="2026-03-31",
        )
        names = {row["feature_name"] for row in rows if row["feature_group"] == "funding_features"}

        self.assertIn("average funding rate across symbols", names)
        self.assertIn("funding dispersion", names)
        self.assertIn("funding sign breadth", names)

    def test_lists_missing_external_features(self) -> None:
        rows = audit_mod.build_feature_rows(
            market_data_complete=True,
            funding_data_complete=True,
            coverage_start="2023-01-01",
            coverage_end="2026-03-31",
        )
        names = {row["feature_name"] for row in rows if row["feature_group"] == "missing_external_features"}

        self.assertIn("open interest", names)
        self.assertIn("long/short ratio", names)
        self.assertIn("taker buy/sell volume", names)
        self.assertIn("premium index / basis", names)
        self.assertIn("mark/index price divergence", names)

    def test_recommends_research_only_when_feature_count_sufficient(self) -> None:
        rows = audit_mod.build_feature_rows(
            market_data_complete=True,
            funding_data_complete=True,
            coverage_start="2023-01-01",
            coverage_end="2026-03-31",
        )

        decision = audit_mod.evaluate_decision(
            rows,
            market_data_complete=True,
            funding_data_complete=True,
            requested_start="2023-01-01",
            requested_end="2026-03-31",
        )

        self.assertTrue(decision["external_regime_classifier_research_allowed"])
        self.assertFalse(decision["strategy_development_allowed"])
        self.assertFalse(decision["demo_live_allowed"])
        self.assertIn("research-only", decision["recommended_next_step"])

    def test_recommends_pause_when_feature_count_insufficient(self) -> None:
        rows = audit_mod.build_feature_rows(
            market_data_complete=False,
            funding_data_complete=False,
            coverage_start="2023-01-01",
            coverage_end="2026-03-31",
        )

        decision = audit_mod.evaluate_decision(
            rows,
            market_data_complete=False,
            funding_data_complete=False,
            requested_start="2023-01-01",
            requested_end="2026-03-31",
        )

        self.assertFalse(decision["external_regime_classifier_research_allowed"])
        self.assertIn("Pause strategy development", decision["recommended_next_step"])
        self.assertTrue(decision["blocking_reasons"])

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payload = self.run_temp_audit(root)
            output_dir = root / "reports" / "research" / "external_regime_feasibility"

            expected_paths = [
                output_dir / "external_regime_feasibility_report.md",
                output_dir / "external_regime_feasibility.json",
                output_dir / "available_features.csv",
                output_dir / "missing_features.csv",
                output_dir / "proposed_regime_features.csv",
            ]
            json_payload = json.loads((output_dir / "external_regime_feasibility.json").read_text(encoding="utf-8"))

            for path in expected_paths:
                self.assertTrue(path.exists(), str(path))
            self.assertFalse(json_payload["decision"]["strategy_development_allowed"])
            self.assertIn("output_paths", payload)

    def test_makefile_target_exists(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("audit-external-regime:", makefile)
        self.assertIn("scripts/audit_external_regime_classifier_feasibility.py", makefile)


if __name__ == "__main__":
    unittest.main()
