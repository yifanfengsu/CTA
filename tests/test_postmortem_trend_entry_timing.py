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

import postmortem_trend_entry_timing as postmortem


FOCUS = "cross_symbol_breadth_acceleration"
OTHER = "pre_breakout_momentum_acceleration"


def trade_row(
    *,
    family: str,
    index: int,
    split: str,
    symbol: str,
    no_cost: float,
    cost_aware: float,
    funding_pnl: float,
) -> dict[str, object]:
    return {
        "event_id": f"{family}_{split}_{index}",
        "family": family,
        "symbol": symbol,
        "timeframe": "4h",
        "split": split,
        "direction": "long" if index % 2 == 0 else "short",
        "test_direction": "long" if index % 2 == 0 else "short",
        "hold_label": "fixed_hold_4h",
        "hold_bars": 1,
        "entry_time": "2025-01-01T00:00:00+08:00",
        "entry_price": 100.0,
        "exit_time": "2025-01-01T04:00:00+08:00",
        "exit_price": 101.0,
        "no_cost_pnl": no_cost,
        "cost_aware_pnl": cost_aware,
        "funding_pnl": funding_pnl,
        "funding_adjusted_pnl": cost_aware + funding_pnl,
        "funding_events_count": 1,
        "funding_data_available": True,
        "funding_interval_covered": True,
        "direction_matches_segment": True,
        "entry_phase": "first_10pct",
        "entry_lag_pct_of_segment": 0.1,
        "missed_mfe_before_entry": 0.1,
        "remaining_mfe_after_entry": 0.5,
        "reverse": False,
    }


def write_required_inputs(research_dir: Path, *, asset_case: bool) -> None:
    research_dir.mkdir(parents=True, exist_ok=True)
    (research_dir / "trend_entry_timing_report.md").write_text("# fixture\n", encoding="utf-8")
    (research_dir / "trend_entry_timing_summary.json").write_text(
        json.dumps(
            {
                "can_enter_entry_timing_phase2": False,
                "stable_like_candidate_exists": False,
                "strategy_development_allowed": False,
                "demo_live_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (research_dir / "data_quality.json").write_text(json.dumps({"market_data_complete": True}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "trade_id": "legacy_1",
                "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "split": "oos_ext",
                "late_entry_flag": True,
            }
        ]
    ).to_csv(research_dir / "legacy_entry_timing_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "event_1",
                "family": FOCUS,
                "symbol": "BTCUSDT_SWAP_OKX.GLOBAL",
                "timeframe": "4h",
                "split": "oos_ext",
                "direction": "long",
            }
        ]
    ).to_csv(research_dir / "candidate_entry_events.csv", index=False)

    focus_direction = 0.70 if asset_case else 0.50
    focus_early = 0.45 if asset_case else 0.10
    focus_largest = 0.50 if asset_case else 0.95
    focus_top = 0.30 if asset_case else 0.95
    focus_reverse = 6.0 if asset_case else 400.0
    focus_random = 12.0 if asset_case else 300.0
    family_summary = pd.DataFrame(
        [
            {
                "family": FOCUS,
                "selected_hold_label": "fixed_hold_4h",
                "event_count": 60,
                "trend_segment_recall": 0.30,
                "early_entry_rate": focus_early,
                "direction_match_rate": focus_direction,
                "median_entry_lag_pct": 0.20 if asset_case else 0.60,
                "average_remaining_mfe": 0.50,
                "average_missed_mfe_before_entry": 0.10,
                "trade_count": 60,
                "no_cost_pnl": 320.0,
                "cost_aware_pnl": 178.0,
                "funding_adjusted_pnl": 195.0,
                "largest_symbol_pnl_share": focus_largest,
                "top_5pct_trade_pnl_contribution": focus_top,
                "train_ext_trade_count": 20,
                "train_ext_no_cost_pnl": 100.0,
                "train_ext_cost_aware_pnl": 80.0,
                "train_ext_funding_adjusted_pnl": 85.0,
                "validation_ext_trade_count": 20,
                "validation_ext_no_cost_pnl": 120.0,
                "validation_ext_cost_aware_pnl": 100.0,
                "validation_ext_funding_adjusted_pnl": 105.0,
                "oos_ext_trade_count": 20,
                "oos_ext_no_cost_pnl": 100.0,
                "oos_ext_cost_aware_pnl": -2.0,
                "oos_ext_funding_adjusted_pnl": 5.0,
                "reverse_test_result": focus_reverse,
                "random_time_control_result": focus_random,
            },
            {
                "family": OTHER,
                "selected_hold_label": "fixed_hold_4h",
                "event_count": 10,
                "trend_segment_recall": 0.10,
                "early_entry_rate": 0.10,
                "direction_match_rate": 0.80,
                "median_entry_lag_pct": 0.70,
                "average_remaining_mfe": 0.20,
                "average_missed_mfe_before_entry": 0.50,
                "trade_count": 15,
                "no_cost_pnl": -30.0,
                "cost_aware_pnl": -45.0,
                "funding_adjusted_pnl": -45.0,
                "largest_symbol_pnl_share": 0.50,
                "top_5pct_trade_pnl_contribution": -1.0,
                "train_ext_trade_count": 5,
                "train_ext_no_cost_pnl": -10.0,
                "train_ext_cost_aware_pnl": -15.0,
                "train_ext_funding_adjusted_pnl": -15.0,
                "validation_ext_trade_count": 5,
                "validation_ext_no_cost_pnl": -10.0,
                "validation_ext_cost_aware_pnl": -15.0,
                "validation_ext_funding_adjusted_pnl": -15.0,
                "oos_ext_trade_count": 5,
                "oos_ext_no_cost_pnl": -10.0,
                "oos_ext_cost_aware_pnl": -15.0,
                "oos_ext_funding_adjusted_pnl": -15.0,
                "reverse_test_result": 1.0,
                "random_time_control_result": 1.0,
            },
        ]
    )
    family_summary.to_csv(research_dir / "candidate_entry_family_summary.csv", index=False)

    symbols = ["BTCUSDT_SWAP_OKX.GLOBAL", "ETHUSDT_SWAP_OKX.GLOBAL"] if asset_case else ["BTCUSDT_SWAP_OKX.GLOBAL"]
    trades: list[dict[str, object]] = []
    for index in range(20):
        trades.append(trade_row(family=FOCUS, index=index, split="train_ext", symbol=symbols[index % len(symbols)], no_cost=5.0, cost_aware=4.0, funding_pnl=0.25))
        trades.append(trade_row(family=FOCUS, index=index, split="validation_ext", symbol=symbols[index % len(symbols)], no_cost=6.0, cost_aware=5.0, funding_pnl=0.25))
        trades.append(trade_row(family=FOCUS, index=index, split="oos_ext", symbol=symbols[index % len(symbols)], no_cost=5.0, cost_aware=-0.10, funding_pnl=0.35))
    for index in range(15):
        split = ["train_ext", "validation_ext", "oos_ext"][index % 3]
        trades.append(trade_row(family=OTHER, index=index, split=split, symbol="BTCUSDT_SWAP_OKX.GLOBAL", no_cost=-2.0, cost_aware=-3.0, funding_pnl=0.0))
    pd.DataFrame(trades).to_csv(research_dir / "candidate_entry_trade_tests.csv", index=False)

    reverse_pnl = 0.10 if asset_case else 20.0
    random_pnl = 0.20 if asset_case else 15.0
    reverse_rows = []
    random_rows = []
    for index in range(60):
        split = ["train_ext", "validation_ext", "oos_ext"][index % 3]
        symbol = symbols[index % len(symbols)]
        reverse_rows.append(trade_row(family=FOCUS, index=index, split=split, symbol=symbol, no_cost=reverse_pnl, cost_aware=reverse_pnl, funding_pnl=0.0))
        random_rows.append(trade_row(family=FOCUS, index=index, split=split, symbol=symbol, no_cost=random_pnl, cost_aware=random_pnl, funding_pnl=0.0))
    pd.DataFrame(reverse_rows).to_csv(research_dir / "candidate_entry_reverse_test.csv", index=False)
    pd.DataFrame(random_rows).to_csv(research_dir / "candidate_entry_random_control.csv", index=False)

    rejected_reasons = (
        "oos_ext:cost_aware_pnl_negative"
        if asset_case
        else "oos_ext:cost_aware_pnl_negative;early_entry_rate_lt_0.40;direction_match_rate_lt_0.55;largest_symbol_pnl_share_gt_0.7;reverse_test_not_clearly_weaker;random_time_control_not_clearly_weaker"
    )
    pd.DataFrame(
        [
            {"family": FOCUS, "stable_like": False, "rejected_reasons": rejected_reasons, "selected_hold_label": "fixed_hold_4h"},
            {"family": OTHER, "stable_like": False, "rejected_reasons": "train_ext:trade_count_lt_10;train_ext:no_cost_pnl_not_positive", "selected_hold_label": "fixed_hold_4h"},
        ]
    ).to_csv(research_dir / "rejected_candidate_entry_families.csv", index=False)

    placeholder = pd.DataFrame([{"family": FOCUS, "placeholder": "ok"}])
    for filename in [
        "candidate_entry_by_symbol.csv",
        "candidate_entry_by_timeframe.csv",
        "candidate_entry_by_split.csv",
        "candidate_entry_concentration.csv",
    ]:
        placeholder.to_csv(research_dir / filename, index=False)


def write_context_dirs(root: Path) -> tuple[Path, Path]:
    trend_map = root / "trend_map"
    trend_map.mkdir()
    (trend_map / "trend_opportunity_summary.json").write_text(
        json.dumps({"enough_trend_opportunities": True, "legacy_analysis": {"main_failure_mode": "entered_middle_or_late"}}),
        encoding="utf-8",
    )
    (trend_map / "data_quality.json").write_text(json.dumps({"all_symbols_complete": True}), encoding="utf-8")
    funding = root / "funding"
    funding.mkdir()
    (funding / "BTC-USDT-SWAP_funding.csv").write_text("funding_time_utc,funding_rate\n", encoding="utf-8")
    return trend_map, funding


class TrendEntryTimingPostmortemTest(unittest.TestCase):
    def run_fixture(self, *, asset_case: bool) -> tuple[dict[str, object], Path]:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        research_dir = root / "research"
        output_dir = root / "postmortem"
        write_required_inputs(research_dir, asset_case=asset_case)
        trend_map, funding = write_context_dirs(root)
        summary = postmortem.run_postmortem(
            research_dir=research_dir,
            trend_map_dir=trend_map,
            funding_dir=funding,
            output_dir=output_dir,
            focus_family=FOCUS,
            timezone_name="Asia/Shanghai",
        )
        return summary, output_dir

    def test_gate_failure_explanation_and_outputs_exist(self) -> None:
        summary, output_dir = self.run_fixture(asset_case=False)

        gate = pd.read_csv(output_dir / "candidate_gate_postmortem.csv")
        other = gate[gate["family"] == OTHER].iloc[0]
        self.assertEqual(other["primary_failure_category"], "no_cost_split_failure")
        self.assertGreaterEqual(int(other["failed_gate_count"]), 2)
        self.assertIn(FOCUS, pd.read_csv(output_dir / "rejected_candidate_entry_postmortem.csv")["family"].tolist())
        for filename in postmortem.OUTPUT_FILES:
            self.assertTrue((output_dir / filename).exists(), filename)
        self.assertFalse(bool(summary["strategy_development_allowed"]))
        self.assertFalse(bool(summary["demo_live_allowed"]))

    def test_focus_deep_dive_cost_sensitivity_and_funding_dependency(self) -> None:
        summary, output_dir = self.run_fixture(asset_case=True)

        deep = pd.read_csv(output_dir / "breadth_candidate_deep_dive.csv").iloc[0]
        self.assertTrue(bool(deep["no_cost_all_splits_positive"]))
        self.assertTrue(bool(deep["funding_dependent"]))
        self.assertTrue(bool(deep["cost_failure_is_marginal"]))

        sensitivity = pd.read_csv(output_dir / "cost_sensitivity.csv")
        self.assertEqual(len(sensitivity.index), 9)
        low_cost = sensitivity[(sensitivity["fee_bps_per_side"] == 2) & (sensitivity["slippage_bps_per_side"] == 2)].iloc[0]
        self.assertTrue(bool(low_cost["passes_cost_aware"]))

        funding = pd.read_csv(output_dir / "funding_dependency.csv")
        oos = funding[funding["scope"] == "oos_ext"].iloc[0]
        self.assertTrue(bool(oos["positive_due_to_funding"]))
        self.assertTrue(bool(summary["funding_dependent"]))

    def test_concentration_and_control_audit(self) -> None:
        _, output_dir = self.run_fixture(asset_case=False)

        concentration = pd.read_csv(output_dir / "entry_timing_concentration_postmortem.csv")
        focus_all = concentration[(concentration["family"] == FOCUS) & (concentration["scope"] == "all_splits")].iloc[0]
        self.assertFalse(bool(focus_all["concentration_pass"]))
        self.assertEqual(int(focus_all["active_symbol_count"]), 1)

        control = pd.read_csv(output_dir / "entry_timing_control_audit.csv")
        all_control = control[control["scope"] == "all_splits"].iloc[0]
        self.assertFalse(bool(all_control["control_pass"]))
        self.assertTrue(bool(all_control["reverse_or_random_stronger"]))

    def test_research_asset_true_case_still_not_strategy_phase2(self) -> None:
        summary, output_dir = self.run_fixture(asset_case=True)

        self.assertTrue(bool(summary["focus_family_research_asset"]))
        self.assertTrue(bool(summary["can_enter_phase1_5_diagnostic"]))
        self.assertFalse(bool(summary["can_enter_entry_timing_phase2"]))
        self.assertFalse(bool(summary["strategy_development_allowed"]))
        self.assertFalse(bool(summary["demo_live_allowed"]))
        self.assertEqual(summary["recommended_next_step"], "phase1_5_execution_and_threshold_diagnostics")
        capture = pd.read_csv(output_dir / "entry_timing_capture_quality.csv")
        self.assertIn("captured_fraction_proxy", capture.columns)

    def test_research_asset_false_case_recommends_pause(self) -> None:
        summary, _ = self.run_fixture(asset_case=False)

        self.assertFalse(bool(summary["focus_family_research_asset"]))
        self.assertFalse(bool(summary["can_enter_phase1_5_diagnostic"]))
        self.assertEqual(summary["recommended_next_step"], "pause_or_new_hypothesis")

    def test_makefile_target_exists(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("postmortem-trend-entry-timing:", text)
        self.assertIn("scripts/postmortem_trend_entry_timing.py", text)


if __name__ == "__main__":
    unittest.main()
