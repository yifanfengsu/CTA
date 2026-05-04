from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_trade_attribution as trade_attr


def create_minimal_report(report_dir: Path) -> None:
    """Create a small report with two round trips."""

    report_dir.mkdir(parents=True, exist_ok=True)
    stats_payload = {
        "statistics_valid": True,
        "bankrupt": False,
        "total_net_pnl": 0.0,
        "engine_trade_count": 4,
        "closed_round_trip_count": 2,
        "gross_profit": 10.0,
        "gross_loss": 10.0,
    }
    trades_df = pd.DataFrame(
        [
            {
                "datetime": "2025-01-01 10:00:00+08:00",
                "vt_tradeid": "BACKTESTING.1",
                "vt_orderid": "BACKTESTING.1",
                "direction": "多",
                "offset": "开",
                "price": 100.0,
                "volume": 1.0,
                "tradeid": "1",
                "orderid": "1",
            },
            {
                "datetime": "2025-01-01 10:05:00+08:00",
                "vt_tradeid": "BACKTESTING.2",
                "vt_orderid": "BACKTESTING.2",
                "direction": "空",
                "offset": "平",
                "price": 110.0,
                "volume": 1.0,
                "tradeid": "2",
                "orderid": "2",
            },
            {
                "datetime": "2025-01-02 11:00:00+08:00",
                "vt_tradeid": "BACKTESTING.3",
                "vt_orderid": "BACKTESTING.3",
                "direction": "空",
                "offset": "开",
                "price": 200.0,
                "volume": 1.0,
                "tradeid": "3",
                "orderid": "3",
            },
            {
                "datetime": "2025-01-02 11:10:00+08:00",
                "vt_tradeid": "BACKTESTING.4",
                "vt_orderid": "BACKTESTING.4",
                "direction": "多",
                "offset": "平",
                "price": 210.0,
                "volume": 1.0,
                "tradeid": "4",
                "orderid": "4",
            },
        ]
    )
    daily_df = pd.DataFrame(
        [
            {"date": "2025-01-01", "net_pnl": 10.0, "trade_count": 2, "balance": 5010.0},
            {"date": "2025-01-02", "net_pnl": -10.0, "trade_count": 2, "balance": 5000.0},
        ]
    )

    (report_dir / "stats.json").write_text(json.dumps(stats_payload, ensure_ascii=False), encoding="utf-8")
    trades_df.to_csv(report_dir / "trades.csv", index=False, encoding="utf-8")
    daily_df.to_csv(report_dir / "daily_pnl.csv", index=False, encoding="utf-8")


class TradeAttributionScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_analyze_trade_attribution")
        self.logger.handlers.clear()

    def test_run_analysis_generates_json_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            report_dir = temp_dir / "minimal_no_cost"
            output_dir = temp_dir / "trade_attribution"
            create_minimal_report(report_dir)

            summary = trade_attr.run_analysis(
                report_dir=report_dir,
                output_dir=output_dir,
                timezone_name="Asia/Shanghai",
                formats={"json", "csv", "md"},
                bar_db_check=False,
                logger=self.logger,
            )

            self.assertEqual(summary["total_trades"], 4)
            self.assertEqual(summary["round_trip_count_inferred_from_trades"], 2)
            self.assertEqual(summary["total_net_pnl"], 0.0)
            self.assertTrue((output_dir / "attribution_summary.json").exists())
            self.assertTrue((output_dir / "attribution_by_side.csv").exists())
            self.assertTrue((output_dir / "attribution_by_hour.csv").exists())
            self.assertTrue((output_dir / "attribution_by_weekday.csv").exists())
            self.assertTrue((output_dir / "attribution_by_month.csv").exists())
            self.assertTrue((output_dir / "attribution_daily_worst.csv").exists())
            self.assertTrue((output_dir / "attribution_report.md").exists())

            summary_payload = json.loads((output_dir / "attribution_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["round_trip_count"], 2)
            self.assertIn("gross_alpha_negative", summary_payload)

            side_df = pd.read_csv(output_dir / "attribution_by_side.csv")
            self.assertEqual(set(side_df["side"]), {"long", "short"})
            markdown_text = (output_dir / "attribution_report.md").read_text(encoding="utf-8")
            self.assertIn("交易归因诊断报告", markdown_text)

    def test_missing_files_warn_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            report_dir = temp_dir / "stats_only"
            output_dir = temp_dir / "out"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "stats.json").write_text(
                json.dumps(
                    {
                        "statistics_valid": True,
                        "bankrupt": False,
                        "total_net_pnl": 1.0,
                        "engine_trade_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            summary = trade_attr.run_analysis(
                report_dir=report_dir,
                output_dir=output_dir,
                timezone_name="Asia/Shanghai",
                formats={"json"},
                bar_db_check=False,
                logger=self.logger,
            )

            warning_text = "\n".join(summary["warnings"])
            self.assertIn("缺少文件: diagnostics.json", warning_text)
            self.assertIn("缺少文件: trades.csv", warning_text)
            self.assertIn("缺少文件: daily_pnl.csv", warning_text)
            self.assertEqual(summary["total_net_pnl"], 1.0)
            self.assertTrue((output_dir / "attribution_summary.json").exists())

    def test_missing_report_dir_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            missing_report_dir = Path(temp_dir_name) / "missing"
            output_dir = Path(temp_dir_name) / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_trade_attribution.py"),
                    "--report-dir",
                    str(missing_report_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
