from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_alpha_diagnostics as alpha_diag


def create_report_dir(
    base_dir: Path,
    name: str,
    total_net_pnl: float,
    rate: float,
    absolute_slippage: float,
) -> Path:
    report_dir = base_dir / name
    report_dir.mkdir(parents=True, exist_ok=True)

    stats_payload = {
        "bankrupt": False,
        "statistics_valid": True,
        "total_net_pnl": total_net_pnl,
        "final_balance": 5000.0 + total_net_pnl,
        "max_ddpercent": -0.5,
        "sharpe_ratio": -1.0,
        "engine_trade_count": 4,
        "closed_round_trip_count": 2,
        "gross_profit": 2.0,
        "gross_loss": 3.0,
    }
    diagnostics_payload = {
        "bankrupt": False,
        "final_balance": 5000.0 + total_net_pnl,
        "min_balance": 4990.0,
        "max_balance": 5005.0,
    }
    run_config_payload = {
        "rate": rate,
        "absolute_slippage": absolute_slippage,
        "strategy_setting": {"contract_size": 1.0},
    }
    daily_df = pd.DataFrame(
        [
            {"date": "2025-01-31", "net_pnl": -1.0, "trade_count": 2, "balance": 4999.0},
            {"date": "2025-02-01", "net_pnl": total_net_pnl + 1.0, "trade_count": 2, "balance": 5000.0 + total_net_pnl},
        ]
    )
    trades_df = pd.DataFrame(
        [
            {"datetime": "2025-01-31 10:00:00+08:00", "vt_tradeid": "1", "tradeid": "1", "orderid": "1", "direction": "多", "offset": "开", "price": 100.0, "volume": 1.0},
            {"datetime": "2025-01-31 10:05:00+08:00", "vt_tradeid": "2", "tradeid": "2", "orderid": "2", "direction": "空", "offset": "平", "price": 99.0, "volume": 1.0},
            {"datetime": "2025-02-01 11:00:00+08:00", "vt_tradeid": "3", "tradeid": "3", "orderid": "3", "direction": "空", "offset": "开", "price": 101.0, "volume": 1.0},
            {"datetime": "2025-02-01 11:15:00+08:00", "vt_tradeid": "4", "tradeid": "4", "orderid": "4", "direction": "多", "offset": "平", "price": 102.0, "volume": 1.0},
        ]
    )
    orders_df = pd.DataFrame(columns=["datetime", "vt_orderid"])

    (report_dir / "stats.json").write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "diagnostics.json").write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "run_config.json").write_text(json.dumps(run_config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    daily_df.to_csv(report_dir / "daily_pnl.csv", index=False, encoding="utf-8")
    trades_df.to_csv(report_dir / "trades.csv", index=False, encoding="utf-8")
    orders_df.to_csv(report_dir / "orders.csv", index=False, encoding="utf-8")
    return report_dir


class AlphaDiagnosticsScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_alpha_diagnostics")
        self.logger.handlers.clear()

    def test_run_analysis_generates_cost_drag_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cost_report_dir = create_report_dir(temp_dir, "cost", total_net_pnl=-5.0, rate=0.0005, absolute_slippage=0.2)
            no_cost_report_dir = create_report_dir(temp_dir, "no_cost", total_net_pnl=-1.0, rate=0.0, absolute_slippage=0.0)
            output_dir = temp_dir / "alpha_output"

            summary = alpha_diag.run_analysis(
                report_dir=cost_report_dir,
                compare_report_dir=no_cost_report_dir,
                output_dir=output_dir,
                top_n=5,
                logger=self.logger,
            )

            self.assertFalse(summary["has_gross_alpha"])
            self.assertIn("无成本版本仍为负收益", summary["alpha_status_text"])

            alpha_summary = json.loads((output_dir / "alpha_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(alpha_summary["cost_impact_summary"]["cost_drag"], 4.0)
            self.assertTrue((output_dir / "monthly_pnl.csv").exists())
            self.assertTrue((output_dir / "weekly_pnl.csv").exists())
            self.assertTrue((output_dir / "trade_side_summary.csv").exists())
            self.assertTrue((output_dir / "trade_duration_summary.csv").exists())

            monthly_df = pd.read_csv(output_dir / "monthly_pnl.csv")
            self.assertGreaterEqual(len(monthly_df.index), 2)

            markdown_text = (output_dir / "alpha_diagnostics.md").read_text(encoding="utf-8")
            self.assertIn("毛 alpha", markdown_text)
            self.assertIn("无成本版本仍为负收益", markdown_text)


if __name__ == "__main__":
    unittest.main()
