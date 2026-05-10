from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_trend_v3_actual_funding as analyze_mod


def write_funding_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["inst_id", "funding_time", "funding_time_utc", "funding_rate", "realized_rate", "raw_funding_rate"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_trade_splits(root: Path) -> None:
    rows_by_split = {
        "train_ext": [
            {
                "policy_name": "v3_1d_ema_50_200_atr5",
                "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "direction": "long",
                "entry_time": "2023-01-01T00:00:00+00:00",
                "exit_time": "2023-01-01T08:00:00+00:00",
                "holding_minutes": 480,
                "entry_price": 100.0,
                "volume": 2.0,
                "contract_size": 1.0,
                "net_pnl": 10.0,
                "no_cost_pnl": 10.0,
                "turnover": 200.0,
            }
        ],
        "validation_ext": [
            {
                "policy_name": "v3_1d_ema_50_200_atr5",
                "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "direction": "short",
                "entry_time": "2023-01-01T08:00:00+00:00",
                "exit_time": "2023-01-01T16:00:00+00:00",
                "holding_minutes": 480,
                "entry_price": 100.0,
                "volume": 2.0,
                "contract_size": 1.0,
                "net_pnl": 10.0,
                "no_cost_pnl": 10.0,
                "turnover": 200.0,
            }
        ],
        "oos_ext": [
            {
                "policy_name": "v3_1d_ema_50_200_atr5",
                "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "direction": "long",
                "entry_time": "2023-01-01T16:00:00+00:00",
                "exit_time": "2023-01-02T00:00:00+00:00",
                "holding_minutes": 480,
                "entry_price": 100.0,
                "volume": 2.0,
                "contract_size": 1.0,
                "net_pnl": 10.0,
                "no_cost_pnl": 10.0,
                "turnover": 200.0,
            }
        ],
    }
    for split, rows in rows_by_split.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(directory / "trend_v3_trades.csv", index=False)
    pd.DataFrame(
        [
            {"policy_name": "v3_1d_ema_50_200_atr5", "net_pnl": 10.0},
        ]
    ).to_csv(root / "oos_ext" / "trend_v3_policy_leaderboard.csv", index=False)


class AnalyzeTrendV3ActualFundingTest(unittest.TestCase):
    def test_conservative_and_signed_funding_calculation(self) -> None:
        self.assertAlmostEqual(analyze_mod.signed_funding_pnl(1000.0, 0.001, "long"), -1.0)
        self.assertAlmostEqual(analyze_mod.signed_funding_pnl(1000.0, 0.001, "short"), 1.0)
        self.assertAlmostEqual(analyze_mod.signed_funding_pnl(1000.0, -0.001, "long"), 1.0)
        self.assertAlmostEqual(analyze_mod.signed_funding_pnl(1000.0, -0.001, "short"), -1.0)

    def test_funding_time_alignment_is_inclusive(self) -> None:
        funding = pd.DataFrame(
            {
                "funding_time_utc": pd.to_datetime(
                    ["2023-01-01T00:00:00Z", "2023-01-01T08:00:00Z", "2023-01-01T16:00:00Z"],
                    utc=True,
                ),
                "funding_rate": [0.001, 0.002, 0.003],
            }
        )
        trade = pd.Series(
            {
                "split": "train_ext",
                "policy_name": "p",
                "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "direction": "long",
                "entry_time": "2023-01-01T00:00:00+00:00",
                "exit_time": "2023-01-01T08:00:00+00:00",
                "holding_minutes": 480,
                "entry_price": 100.0,
                "volume": 1.0,
                "contract_size": 1.0,
                "net_pnl": 5.0,
                "no_cost_pnl": 5.0,
            }
        )

        row = analyze_mod.analyze_trade_row(trade, {"BTC-USDT-SWAP": funding}, set(), "UTC")

        self.assertEqual(row["funding_events_count"], 2)
        self.assertAlmostEqual(row["conservative_funding_cost"], 0.3)
        self.assertAlmostEqual(row["signed_funding_pnl"], -0.3)

    def test_missing_funding_csv_warns_without_crashing(self) -> None:
        trade = pd.Series(
            {
                "split": "train_ext",
                "policy_name": "p",
                "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "direction": "long",
                "entry_time": "2023-01-01T00:00:00+00:00",
                "exit_time": "2023-01-01T08:00:00+00:00",
                "entry_price": 100.0,
                "volume": 1.0,
                "contract_size": 1.0,
                "net_pnl": 5.0,
            }
        )

        row = analyze_mod.analyze_trade_row(trade, {}, {"BTC-USDT-SWAP"}, "UTC")

        self.assertEqual(row["funding_events_count"], 0)
        self.assertIn("missing_funding_csv_or_empty_history", row["warnings"])

    def test_run_analysis_writes_policy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            funding_dir = root / "funding"
            trend_dir = root / "trend"
            compare_dir = root / "compare"
            output_dir = root / "output"
            write_trade_splits(trend_dir)
            write_funding_csv(
                funding_dir / "BTC-USDT-SWAP_funding_2023-01-01_2023-01-02.csv",
                [
                    {
                        "inst_id": "BTC-USDT-SWAP",
                        "funding_time": "1672531200000",
                        "funding_time_utc": "2023-01-01T00:00:00+00:00",
                        "funding_rate": "0.001",
                        "realized_rate": "0.001",
                        "raw_funding_rate": "0.001",
                    },
                    {
                        "inst_id": "BTC-USDT-SWAP",
                        "funding_time": "1672560000000",
                        "funding_time_utc": "2023-01-01T08:00:00+00:00",
                        "funding_rate": "0.001",
                        "realized_rate": "0.001",
                        "raw_funding_rate": "0.001",
                    },
                    {
                        "inst_id": "BTC-USDT-SWAP",
                        "funding_time": "1672588800000",
                        "funding_time_utc": "2023-01-01T16:00:00+00:00",
                        "funding_rate": "-0.001",
                        "realized_rate": "-0.001",
                        "raw_funding_rate": "-0.001",
                    },
                ],
            )
            compare_dir.mkdir(parents=True, exist_ok=True)
            (compare_dir / "trend_v3_extended_compare_summary.json").write_text(
                json.dumps({"stable_candidates": [{"policy_name": "v3_1d_ema_50_200_atr5"}]}),
                encoding="utf-8",
            )

            summary = analyze_mod.run_analysis(
                funding_dir=funding_dir,
                trend_v3_extended_dir=trend_dir,
                compare_dir=compare_dir,
                output_dir=output_dir,
                timezone_name="UTC",
                mode="conservative",
                inst_ids=["BTC-USDT-SWAP"],
            )

            self.assertTrue((output_dir / "actual_funding_policy_summary.csv").exists())
            self.assertTrue((output_dir / "actual_funding_report.md").exists())
            self.assertIn("v3_1d_ema_50_200_atr5", summary["funding_adjusted_all_split_positive_policies_conservative"])

    def test_incomplete_funding_forces_final_gates_false_and_explains_zero_events(self) -> None:
        policy_summary = pd.DataFrame(
            [
                {
                    "policy_name": "v3_1d_ema_50_200_atr5",
                    "split": split,
                    "original_net_pnl": 10.0,
                    "funding_adjusted_net_pnl_conservative": 9.0,
                    "funding_adjusted_net_pnl_signed": 9.0,
                    "funding_events_count": 0,
                }
                for split in analyze_mod.SPLITS
            ]
        )
        split_summary = pd.DataFrame(
            [{"split": split, "funding_events_count": 0, "original_net_pnl": 10.0} for split in analyze_mod.SPLITS]
        )

        summary = analyze_mod.build_summary(
            adjustments=pd.DataFrame(),
            policy_summary=policy_summary,
            split_summary=split_summary,
            symbol_summary=pd.DataFrame(),
            funding_missing_inst_ids=[],
            funding_warnings=[],
            verify_summary={"funding_data_complete": False, "incomplete_reason": ["missing_before_first_available"]},
            compare_summary={"stable_candidates": [{"policy_name": "v3_1d_ema_50_200_atr5"}]},
            trend_v3_extended_dir=Path("/tmp/not-used"),
            funding_dir=Path("/tmp/not-used"),
            output_dir=Path("/tmp/not-used"),
            mode="conservative",
        )

        self.assertFalse(summary["funding_data_complete"])
        self.assertFalse(summary["funding_adjusted_stable_candidate_exists"])
        self.assertFalse(summary["can_enter_funding_aware_v3_1_research"])
        self.assertFalse(summary["strategy_development_allowed"])
        self.assertFalse(summary["demo_live_allowed"])
        self.assertEqual(summary["funding_event_coverage_warning"], "likely_due_to_missing_funding_coverage")
        self.assertIn("train_ext", summary["zero_funding_event_splits_when_incomplete"])

        report = analyze_mod.render_report(summary)
        self.assertIn("available_data_only=true", report.splitlines()[3])
        self.assertIn("likely_due_to_missing_funding_coverage", report)

    def test_makefile_targets_exist(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        for target in [
            "download-funding-dry-run:",
            "download-funding:",
            "verify-funding:",
            "verify-funding-allow-partial:",
            "import-funding-csv:",
            "analyze-trend-v3-funding:",
        ]:
            self.assertIn(target, text)


if __name__ == "__main__":
    unittest.main()
