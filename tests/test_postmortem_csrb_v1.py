from __future__ import annotations

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

import postmortem_csrb_v1 as postmortem


def write_required_placeholders(research_dir: Path) -> None:
    for filename in [
        "event_group_summary.csv",
        "trade_group_summary.csv",
        "by_symbol.csv",
        "by_timeframe.csv",
        "by_split.csv",
        "session_summary.csv",
        "concentration.csv",
        "reverse_test.csv",
        "random_time_control.csv",
    ]:
        pd.DataFrame([{"placeholder": "ok"}]).to_csv(research_dir / filename, index=False)
    pd.DataFrame(
        [
            {
                "inst_id": "BTC-USDT-SWAP",
                "funding_data_complete": True,
                "funding_pnl": 0.0,
            }
        ]
    ).to_csv(research_dir / "funding_summary.csv", index=False)
    (research_dir / "summary.md").write_text("# CSRB fixture\n", encoding="utf-8")


def write_fixture(research_dir: Path) -> None:
    research_dir.mkdir(parents=True, exist_ok=True)
    split_dates = {
        "train_start": "2025-01-01T00:00:00+00:00",
        "train_end": "2025-01-02T00:00:00+00:00",
        "validation_start": "2025-01-02T00:00:00+00:00",
        "validation_end": "2025-01-03T00:00:00+00:00",
        "oos_start": "2025-01-03T00:00:00+00:00",
        "oos_end": "2025-01-04T00:00:00+00:00",
    }
    summary = {
        "final_decision": "postmortem",
        "continue_to_phase2": False,
        "train_pass": False,
        "validation_pass": False,
        "oos_pass": False,
        "cost_aware_pass": False,
        "funding_adjusted_pass": False,
        "trade_count_pass": True,
        "concentration_pass": False,
        "reverse_test_pass": False,
        "session_vs_baseline_pass": False,
        "strategy_development_allowed": False,
        "demo_live_allowed": False,
        "split_dates": split_dates,
        "gates": {
            "train_pass": False,
            "validation_pass": False,
            "oos_pass": False,
            "cost_aware_pass": False,
            "funding_adjusted_pass": False,
            "trade_count_pass": True,
            "concentration_pass": False,
            "reverse_test_pass": False,
            "session_vs_baseline_pass": False,
            "continue_to_phase2": False,
        },
        "warnings": [],
    }
    (research_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    data_quality = {
        "all_market_data_complete": True,
        "funding": {"funding_data_complete": True, "records": {}},
    }
    (research_dir / "data_quality.json").write_text(json.dumps(data_quality), encoding="utf-8")

    events = []
    trades = []
    source_specs = [
        ("train", "2025-01-01", "B", "asia_to_europe", "long", -10.0, 10.0, 9, 8),
        ("validation", "2025-01-02", "C", "europe_to_us", "short", -8.0, 8.0, 8, 7),
        ("oos", "2025-01-03", "B", "asia_to_europe", "long", -6.0, 6.0, 7, 6),
    ]
    for index, (split, date, group, session_type, direction, forward_pnl, reverse_pnl, random_pnl, random_cost_pnl) in enumerate(source_specs, start=1):
        if session_type == "asia_to_europe":
            timestamp = f"{date}T08:14:00+00:00"
            random_timestamp = f"{date}T01:14:00+00:00"
        else:
            timestamp = f"{date}T13:14:00+00:00"
            random_timestamp = f"{date}T03:14:00+00:00"
        reverse_direction = "short" if direction == "long" else "long"
        for event_id, event_group, event_session_type, source_session_type, event_direction, event_timestamp, pnl in [
            (f"core_{index}", group, session_type, session_type, direction, timestamp, forward_pnl),
            (f"reverse_{index}", "E", "reverse_test", session_type, reverse_direction, timestamp, reverse_pnl),
            (f"random_{index}", "D", "random_time_control", session_type, direction, random_timestamp, random_pnl),
        ]:
            events.append(
                {
                    "event_id": event_id,
                    "timestamp": event_timestamp,
                    "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                    "timeframe": "15m",
                    "group": event_group,
                    "session_type": event_session_type,
                    "source_session_type": source_session_type,
                    "session_date": date,
                    "direction": event_direction,
                    "range_width": 10.0,
                    "atr_prev": 2.0,
                    "breakout_boundary": 105.0,
                    "range_high": 104.5,
                    "range_low": 95.0,
                    "close": 106.0 if event_direction == "long" else 94.0,
                    "entry_time": f"{date}T08:15:00+00:00",
                    "entry_price": 100.0,
                    "future_return_4": pnl / 1000.0,
                    "future_return_8": pnl / 1000.0,
                    "future_return_16": pnl / 1000.0,
                    "future_return_32": pnl / 1000.0,
                    "control_key": "k" if event_group == "D" else "",
                }
            )
            cost_pnl = pnl - 2.0
            funding_pnl = cost_pnl
            if event_group == "D":
                cost_pnl = random_cost_pnl
                funding_pnl = random_cost_pnl
            trades.append(
                {
                    "trade_id": f"trade_{event_id}",
                    "event_id": event_id,
                    "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                    "inst_id": "BTC-USDT-SWAP",
                    "timeframe": "15m",
                    "group": event_group,
                    "session_type": event_session_type,
                    "source_session_type": source_session_type,
                    "session_date": date,
                    "direction": event_direction,
                    "entry_time": f"{date}T08:15:00+00:00",
                    "entry_price": 100.0,
                    "exit_time": f"{date}T12:15:00+00:00",
                    "exit_price": 99.0,
                    "hold_bars": 16,
                    "gross_return": pnl / 1000.0,
                    "no_cost_pnl": pnl,
                    "fee_cost": 1.0,
                    "slippage_cost": 1.0,
                    "funding_count": 0,
                    "funding_pnl": 0.0,
                    "cost_aware_pnl": cost_pnl,
                    "funding_adjusted_pnl": funding_pnl,
                    "split": split,
                }
            )
    pd.DataFrame(events).to_csv(research_dir / "events.csv", index=False)
    pd.DataFrame(trades).to_csv(research_dir / "trades.csv", index=False)
    write_required_placeholders(research_dir)


class PostmortemCsrbV1Test(unittest.TestCase):
    def test_missing_files_warning_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            research_dir = root / "research"
            output_dir = root / "postmortem"
            research_dir.mkdir()
            final = postmortem.run_postmortem(research_dir=research_dir, output_dir=output_dir)

            self.assertTrue(final["warnings"])
            self.assertTrue((output_dir / "csrb_v1_postmortem_summary.json").exists())

    def test_full_postmortem_outputs_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            research_dir = root / "research"
            output_dir = root / "postmortem"
            write_fixture(research_dir)

            final = postmortem.run_postmortem(research_dir=research_dir, output_dir=output_dir)

            self.assertTrue(final["csrb_v1_failed"])
            self.assertFalse(final["continue_to_phase2"])
            self.assertFalse(final["strategy_development_allowed"])
            self.assertFalse(final["demo_live_allowed"])
            self.assertTrue(final["possible_false_breakout_research_hypothesis"])
            self.assertTrue(final["random_control_requires_review"])
            self.assertFalse(final["gates"]["train_pass"])
            self.assertFalse(final["gates"]["validation_pass"])
            self.assertFalse(final["gates"]["oos_pass"])
            self.assertFalse(final["gates"]["cost_aware_pass"])
            self.assertFalse(final["gates"]["funding_adjusted_pass"])
            self.assertFalse(final["gates"]["continue_to_phase2"])

            required = [
                "csrb_v1_postmortem_report.md",
                "csrb_v1_postmortem_summary.json",
                "implementation_sanity.csv",
                "session_failure_decomposition.csv",
                "random_control_audit.csv",
                "random_control_seed_robustness.csv",
                "reverse_directionality_postmortem.csv",
                "postmortem_by_symbol.csv",
                "postmortem_by_direction.csv",
                "postmortem_by_timeframe.csv",
                "postmortem_by_session.csv",
                "postmortem_by_symbol_direction.csv",
                "postmortem_by_symbol_session.csv",
                "horizon_path_postmortem.csv",
                "feature_bin_postmortem.csv",
                "conflict_filter_impact.csv",
            ]
            for filename in required:
                self.assertTrue((output_dir / filename).exists(), filename)

            seed_rows = pd.read_csv(output_dir / "random_control_seed_robustness.csv")
            feature_rows = pd.read_csv(output_dir / "feature_bin_postmortem.csv")
            conflict_rows = pd.read_csv(output_dir / "conflict_filter_impact.csv")
            self.assertFalse(seed_rows.empty)
            self.assertFalse(feature_rows.empty)
            self.assertFalse(conflict_rows.empty)

    def test_makefile_target_exists(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("postmortem-csrb-v1:", text)
        self.assertIn("scripts/postmortem_csrb_v1.py", text)


if __name__ == "__main__":
    unittest.main()
