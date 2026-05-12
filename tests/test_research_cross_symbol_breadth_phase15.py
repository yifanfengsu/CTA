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

import research_cross_symbol_breadth_phase15 as phase15


FOCUS = "cross_symbol_breadth_acceleration"


def make_trade(
    *,
    index: int,
    split: str,
    symbol: str = "AAA",
    no_cost: float = 10.0,
    cost: float = 8.0,
    funding: float = 9.0,
    direction: str = "long",
    reverse: bool = False,
) -> dict[str, object]:
    return {
        "event_id": f"event_{split}_{index}",
        "family": FOCUS,
        "symbol": symbol,
        "timeframe": "4h",
        "split": split,
        "direction": direction,
        "test_direction": direction,
        "hold_label": "fixed_hold_4h",
        "no_cost_pnl": no_cost,
        "cost_aware_pnl": cost,
        "funding_pnl": funding - cost,
        "funding_adjusted_pnl": funding,
        "direction_matches_segment": True,
        "entry_phase": "first_10pct" if index % 2 == 0 else "middle_25_75pct",
        "entry_lag_pct_of_segment": 0.1 if index % 2 == 0 else 0.5,
        "trend_segment_id": f"seg_{index // 2}",
        "reverse": reverse,
    }


def make_event(index: int, split: str, *, symbol: str = "AAA", direction: str = "long") -> dict[str, object]:
    return {
        "event_id": f"event_{split}_{index}",
        "family": FOCUS,
        "symbol": symbol,
        "timeframe": "4h",
        "split": split,
        "direction": direction,
        "event_time": f"2025-01-{(index % 20) + 1:02d}T00:00:00+08:00",
        "ret_3": 0.01 + index * 0.001,
        "ret_6": 0.02 + index * 0.001,
        "ret_20": 0.03 + index * 0.001,
        "funding_rate": 0.0001,
        "entry_price_vs_segment_start_price": 0.1,
        "direction_matches_segment": True,
        "entry_phase": "first_10pct" if index % 2 == 0 else "middle_25_75pct",
        "entry_lag_pct_of_segment": 0.1 if index % 2 == 0 else 0.5,
        "trend_segment_id": f"seg_{index // 2}",
    }


def family_summary_row() -> dict[str, object]:
    row: dict[str, object] = {
        "family": FOCUS,
        "selected_hold_label": "fixed_hold_4h",
        "trend_segment_recall": 0.3,
        "early_entry_rate": 0.4,
        "direction_match_rate": 0.8,
        "largest_symbol_pnl_share": 0.5,
        "top_5pct_trade_pnl_contribution": 0.5,
    }
    for split in phase15.SPLITS:
        row[f"{split}_trade_count"] = 10
        row[f"{split}_no_cost_pnl"] = 100.0
        row[f"{split}_cost_aware_pnl"] = 80.0
        row[f"{split}_funding_adjusted_pnl"] = 90.0
    return row


class CrossSymbolBreadthPhase15Test(unittest.TestCase):
    def test_cost_edge_margin_diagnostic_and_break_even(self) -> None:
        trades = pd.DataFrame([make_trade(index=index, split="oos_ext", no_cost=10.0, cost=9.0, funding=9.5) for index in range(10)])

        out = phase15.build_cost_edge_margin(trades).iloc[0]

        self.assertAlmostEqual(out["gross_edge_before_cost"], 100.0)
        self.assertAlmostEqual(out["total_fee_cost"], 5.0)
        self.assertAlmostEqual(out["total_slippage_cost"], 5.0)
        self.assertAlmostEqual(out["break_even_fee_bps"], 50.0)
        self.assertAlmostEqual(out["break_even_slippage_bps"], 50.0)
        self.assertAlmostEqual(out["break_even_equal_fee_slippage_bps"], 25.0)

    def test_signal_strength_bucket_output(self) -> None:
        events = pd.DataFrame([make_event(index=index, split=["train_ext", "validation_ext", "oos_ext"][index % 3]) for index in range(30)])
        trades = pd.DataFrame([make_trade(index=index, split=["train_ext", "validation_ext", "oos_ext"][index % 3]) for index in range(30)])
        warnings: list[str] = []

        out = phase15.build_signal_strength_buckets(events, trades, warnings)

        self.assertFalse(out.empty)
        self.assertIn("feature", out.columns)
        self.assertIn("signal_strength_requested_fields_missing_using_fallback:ret_3,ret_6,ret_20,funding_rate,entry_price_vs_segment_start_price", warnings)

    def test_concentration_repair_and_top_removal(self) -> None:
        trades = pd.DataFrame(
            [
                make_trade(index=0, split="train_ext", symbol="AAA", no_cost=100.0, cost=100.0, funding=100.0),
                make_trade(index=1, split="train_ext", symbol="AAA", no_cost=90.0, cost=90.0, funding=90.0),
                make_trade(index=2, split="train_ext", symbol="BBB", no_cost=10.0, cost=10.0, funding=10.0),
            ]
        )

        out = phase15.build_concentration_repair(trades)
        remove_top_1 = out[(out["scope"] == "all_splits") & (out["repair_action"] == "remove_top_1_trade")].iloc[0]
        remove_top_5 = out[(out["scope"] == "all_splits") & (out["repair_action"] == "remove_top_5pct_trades")].iloc[0]
        remove_symbol = out[(out["scope"] == "all_splits") & (out["repair_action"] == "remove_largest_symbol")].iloc[0]

        self.assertAlmostEqual(remove_top_1["funding_adjusted_pnl"], 100.0)
        self.assertAlmostEqual(remove_top_5["funding_adjusted_pnl"], 100.0)
        self.assertAlmostEqual(remove_symbol["funding_adjusted_pnl"], 10.0)

    def test_funding_dependency_decision(self) -> None:
        trades = pd.DataFrame([make_trade(index=index, split="oos_ext", no_cost=10.0, cost=-1.0, funding=2.0) for index in range(3)])

        out = phase15.build_funding_dependency(trades, pd.DataFrame())
        oos = out[out["scope"] == "oos_ext"].iloc[0]

        self.assertTrue(bool(oos["positive_due_to_funding"]))
        self.assertTrue(bool(oos["funding_carry_contaminated"]))

    def test_early_entry_improvement_decision(self) -> None:
        events = []
        trades = []
        for index in range(20):
            split = ["train_ext", "validation_ext", "oos_ext"][index % 3]
            events.append(make_event(index=index, split=split))
            trades.append(make_trade(index=index, split=split, no_cost=10.0, cost=9.0, funding=9.0))
        out = phase15.build_early_entry_improvement(pd.DataFrame(events), pd.DataFrame(trades), [])

        first = out[out["diagnostic_group"] == "first_breadth_acceleration_event"].iloc[0]

        self.assertIn("early_entry_rate", out.columns)
        self.assertGreaterEqual(first["early_entry_rate"], 0.0)

    def test_random_control_robustness(self) -> None:
        family_summary = pd.DataFrame([family_summary_row()])
        forward = pd.DataFrame([make_trade(index=index, split="oos_ext", funding=10.0) for index in range(10)])
        reverse = pd.DataFrame([make_trade(index=index, split="oos_ext", funding=-10.0, reverse=True) for index in range(10)])
        random = pd.DataFrame([make_trade(index=index, split="oos_ext", funding=-5.0) for index in range(10)])

        out = phase15.build_control_robustness(forward, reverse, random, family_summary, FOCUS, seeds=20)
        oos = out[out["scope"] == "oos_ext"].iloc[0]

        self.assertTrue(bool(oos["reverse_weaker"]))
        self.assertTrue(bool(oos["significantly_beats_random"]))

    def test_research_asset_true_case(self) -> None:
        cost = pd.DataFrame([{"cost_failure_is_marginal": True, "break_even_equal_fee_slippage_bps": 2.5}])
        concentration = pd.DataFrame(
            [
                {"scope": "all_splits", "repair_action": "cap_single_symbol_pnl_share", "non_disastrous": True, "no_cost_pnl": 10.0, "funding_adjusted_pnl": 1.0},
                {"scope": "all_splits", "repair_action": "remove_top_1_trade", "non_disastrous": True, "no_cost_pnl": 10.0, "funding_adjusted_pnl": 1.0},
                {"scope": "all_splits", "repair_action": "remove_top_5pct_trades", "non_disastrous": True, "no_cost_pnl": 10.0, "funding_adjusted_pnl": 1.0},
            ]
        )
        funding = pd.DataFrame([{"scope": "oos_ext", "funding_carry_contaminated": False}])
        early = pd.DataFrame([{"diagnostic_group": "first_breadth_acceleration_event", "early_entry_rate": 0.36, "direction_and_pnl_not_sacrificed": True}])
        control = pd.DataFrame([{"scope": "all_splits", "control_robust": True}])

        rejected, asset = phase15.phase15_reasons(
            cost_edge=cost,
            concentration=concentration,
            funding=funding,
            early_entry=early,
            control=control,
            family_row=pd.Series(family_summary_row()),
        )

        self.assertTrue(asset)
        self.assertTrue(rejected["passed"].all())

    def test_research_asset_false_case(self) -> None:
        rejected, asset = phase15.phase15_reasons(
            cost_edge=pd.DataFrame([{"cost_failure_is_marginal": False, "break_even_equal_fee_slippage_bps": 1.0}]),
            concentration=pd.DataFrame(),
            funding=pd.DataFrame([{"scope": "oos_ext", "funding_carry_contaminated": True}]),
            early_entry=pd.DataFrame(),
            control=pd.DataFrame([{"scope": "all_splits", "control_robust": False}]),
            family_row=pd.Series(family_summary_row()),
        )

        self.assertFalse(asset)
        self.assertIn(False, rejected["passed"].tolist())

    def write_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        entry = root / "entry"
        post = root / "post"
        trend = root / "trend"
        funding = root / "funding"
        entry.mkdir()
        post.mkdir()
        trend.mkdir()
        funding.mkdir()
        events = []
        trades = []
        for index in range(30):
            split = ["train_ext", "validation_ext", "oos_ext"][index % 3]
            symbol = ["AAA", "BBB", "CCC"][index % 3]
            events.append(make_event(index=index, split=split, symbol=symbol))
            trades.append(make_trade(index=index, split=split, symbol=symbol, no_cost=10.0, cost=8.0, funding=9.0))
        pd.DataFrame(events).to_csv(entry / "candidate_entry_events.csv", index=False)
        pd.DataFrame(trades).to_csv(entry / "candidate_entry_trade_tests.csv", index=False)
        pd.DataFrame([family_summary_row()]).to_csv(entry / "candidate_entry_family_summary.csv", index=False)
        pd.DataFrame([{"placeholder": "ok"}]).to_csv(entry / "candidate_entry_concentration.csv", index=False)
        pd.DataFrame([make_trade(index=index, split=["train_ext", "validation_ext", "oos_ext"][index % 3], funding=-2.0, reverse=True) for index in range(30)]).to_csv(entry / "candidate_entry_reverse_test.csv", index=False)
        pd.DataFrame([make_trade(index=index, split=["train_ext", "validation_ext", "oos_ext"][index % 3], funding=-1.0) for index in range(30)]).to_csv(entry / "candidate_entry_random_control.csv", index=False)

        for filename in [
            "breadth_candidate_deep_dive.csv",
            "cost_sensitivity.csv",
            "funding_dependency.csv",
            "entry_timing_concentration_postmortem.csv",
            "entry_timing_control_audit.csv",
            "entry_timing_capture_quality.csv",
        ]:
            pd.DataFrame([{"family": FOCUS, "placeholder": "ok"}]).to_csv(post / filename, index=False)
        (trend / "trend_opportunity_summary.json").write_text(json.dumps({"enough_trend_opportunities": True, "legacy_analysis": {"main_failure_mode": "entered_middle_or_late"}}), encoding="utf-8")
        (trend / "data_quality.json").write_text(json.dumps({"all_symbols_complete": True}), encoding="utf-8")
        return entry, post, trend, funding

    def test_output_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entry, post, trend, funding = self.write_fixture(root)
            output = root / "out"

            summary = phase15.run_research(
                entry_timing_dir=entry,
                postmortem_dir=post,
                trend_map_dir=trend,
                funding_dir=funding,
                output_dir=output,
                focus_family=FOCUS,
                timezone_name="Asia/Shanghai",
                logger=logging.getLogger("test_phase15"),
            )

            self.assertIn("research_asset", summary)
            for filename in phase15.OUTPUT_FILES:
                self.assertTrue((output / filename).exists(), filename)

    def test_makefile_target_exists(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-breadth-phase15:", text)
        self.assertIn("scripts/research_cross_symbol_breadth_phase15.py", text)


if __name__ == "__main__":
    unittest.main()
