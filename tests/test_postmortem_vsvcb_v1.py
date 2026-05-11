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

import postmortem_vsvcb_v1 as postmortem


SYMBOLS = [
    "BTCUSDT_SWAP_OKX.GLOBAL",
    "ETHUSDT_SWAP_OKX.GLOBAL",
    "SOLUSDT_SWAP_OKX.GLOBAL",
]


def split_timestamp(split: str, index: int) -> pd.Timestamp:
    starts = {
        "train": pd.Timestamp("2025-01-02T00:00:00+08:00"),
        "validation": pd.Timestamp("2025-02-02T00:00:00+08:00"),
        "oos": pd.Timestamp("2025-03-02T00:00:00+08:00"),
    }
    return starts[split] + pd.Timedelta(hours=index)


def make_event(
    *,
    event_id: str,
    timestamp: pd.Timestamp,
    group: str,
    direction: str,
    symbol: str,
    pnl_sign: float,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "timeframe": "15m",
        "direction": direction,
        "group": group,
        "close": 100.0,
        "open_next": 100.0,
        "breakout_boundary": 99.0,
        "bb_width": 0.01,
        "bb_width_percentile": 0.12,
        "squeeze": True,
        "volume": 200.0,
        "volume_ma_prev": 100.0,
        "volume_ratio": 2.0,
        "volume_confirm": True,
        "atr": 1.5,
        "entry_close_theoretical": 100.0,
        "entry_next_open": 100.0,
        "future_return_3": pnl_sign * 0.003,
        "future_return_5": pnl_sign * 0.004,
        "future_return_10": pnl_sign * 0.005,
        "future_return_20": pnl_sign * 0.006,
        "mfe_10": 0.004 if pnl_sign < 0 else 0.008,
        "mae_10": -0.008 if pnl_sign < 0 else -0.004,
        "mfe_mae_ratio_10": 0.5 if pnl_sign < 0 else 2.0,
        "reversal_flag_3": pnl_sign < 0,
        "reversal_flag_5": pnl_sign < 0,
        "reversal_flag_10": pnl_sign < 0,
        "funding_crossed": False,
        "funding_cost_estimate": 0.0,
        "bar_index": 1,
        "inst_id": "BTC-USDT-SWAP",
    }


def make_trade(
    *,
    trade_id: str,
    event_id: str,
    timestamp: pd.Timestamp,
    group: str,
    direction: str,
    symbol: str,
    split: str,
    no_cost_pnl: float,
) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "event_id": event_id,
        "symbol": symbol,
        "timeframe": "15m",
        "group": group,
        "direction": direction,
        "entry_time": (timestamp + pd.Timedelta(minutes=1)).isoformat(),
        "entry_price": 100.0,
        "exit_time": (timestamp + pd.Timedelta(minutes=16)).isoformat(),
        "exit_price": 101.0,
        "hold_bars": 10,
        "gross_return": no_cost_pnl / 1000.0,
        "no_cost_pnl": no_cost_pnl,
        "fee_cost": 0.25,
        "slippage_cost": 0.25,
        "funding_pnl": 0.0,
        "cost_aware_pnl": no_cost_pnl - 0.5,
        "funding_adjusted_pnl": no_cost_pnl - 0.5,
        "mfe": 0.004,
        "mae": -0.008,
        "mfe_mae_ratio": 0.5,
        "reversal_flags": "3:True;5:True;10:True",
        "split": split,
    }


def write_fixture(research_dir: Path, *, complete_inputs: bool = True) -> None:
    research_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    trade_index = 1
    counts = {"train": 30, "validation": 10, "oos": 10}
    for split, count in counts.items():
        for index in range(count):
            timestamp = split_timestamp(split, index)
            symbol = SYMBOLS[index % len(SYMBOLS)]
            direction = "long" if index % 2 == 0 else "short"
            reverse_direction = "short" if direction == "long" else "long"
            d_event_id = f"d_{split}_{index}"
            e_event_id = f"e_{split}_{index}"
            events.append(
                make_event(
                    event_id=d_event_id,
                    timestamp=timestamp,
                    group="D",
                    direction=direction,
                    symbol=symbol,
                    pnl_sign=-1.0,
                )
            )
            events.append(
                make_event(
                    event_id=e_event_id,
                    timestamp=timestamp,
                    group="E",
                    direction=reverse_direction,
                    symbol=symbol,
                    pnl_sign=1.0,
                )
            )
            for group, baseline_pnl in [("A", -3.0), ("B", -2.0), ("C", -4.0)]:
                event_id = f"{group.lower()}_{split}_{index}"
                events.append(
                    make_event(
                        event_id=event_id,
                        timestamp=timestamp,
                        group=group,
                        direction=direction,
                        symbol=symbol,
                        pnl_sign=-1.0,
                    )
                )
                if split == "oos":
                    trades.append(
                        make_trade(
                            trade_id=f"t_{trade_index}",
                            event_id=event_id,
                            timestamp=timestamp,
                            group=group,
                            direction=direction,
                            symbol=symbol,
                            split=split,
                            no_cost_pnl=baseline_pnl,
                        )
                    )
                    trade_index += 1
            trades.append(
                make_trade(
                    trade_id=f"t_{trade_index}",
                    event_id=d_event_id,
                    timestamp=timestamp,
                    group="D",
                    direction=direction,
                    symbol=symbol,
                    split=split,
                    no_cost_pnl=-1.0,
                )
            )
            trade_index += 1
            trades.append(
                make_trade(
                    trade_id=f"t_{trade_index}",
                    event_id=e_event_id,
                    timestamp=timestamp,
                    group="E",
                    direction=reverse_direction,
                    symbol=symbol,
                    split=split,
                    no_cost_pnl=1.0,
                )
            )
            trade_index += 1

    pd.DataFrame(events).to_csv(research_dir / "events.csv", index=False)
    pd.DataFrame(trades).to_csv(research_dir / "trades.csv", index=False)
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
        "ablation_pass": True,
        "funding_data_complete": True,
        "split_dates": {
            "train_start": "2025-01-01T00:00:00+08:00",
            "train_end": "2025-02-01T00:00:00+08:00",
            "validation_start": "2025-02-01T00:00:00+08:00",
            "validation_end": "2025-03-01T00:00:00+08:00",
            "oos_start": "2025-03-01T00:00:00+08:00",
            "oos_end": "2025-04-01T00:00:00+08:00",
        },
        "event_counts": {"A": 50, "B": 50, "C": 50, "D": 50, "E": 50},
        "trade_counts": {"A": 10, "B": 10, "C": 10, "D": 50, "E": 50},
        "gates": {
            "train_no_cost_pnl": -30.0,
            "validation_no_cost_pnl": -10.0,
            "oos_no_cost_pnl": -10.0,
            "oos_cost_aware_pnl": -15.0,
            "oos_funding_adjusted_pnl": -15.0,
        },
        "warnings": ["skipped_events_due_to_single_position_filter=0"],
    }
    (research_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    data_quality = {
        "all_market_data_complete": True,
        "funding": {"funding_data_complete": True, "records": {}},
    }
    (research_dir / "data_quality.json").write_text(json.dumps(data_quality), encoding="utf-8")
    if not complete_inputs:
        return

    reverse_rows = []
    for split, d_pnl in [("train", -30.0), ("validation", -10.0), ("oos", -10.0), ("all", -50.0)]:
        count = 50 if split == "all" else counts[split]
        reverse_rows.append(
            {
                "timeframe": "15m",
                "split": split,
                "d_trade_count": count,
                "e_trade_count": count,
                "d_no_cost_pnl": d_pnl,
                "e_no_cost_pnl": -d_pnl,
                "d_cost_aware_pnl": d_pnl - count * 0.5,
                "e_cost_aware_pnl": -d_pnl - count * 0.5,
                "d_funding_adjusted_pnl": d_pnl - count * 0.5,
                "e_funding_adjusted_pnl": -d_pnl - count * 0.5,
                "reverse_weaker": False,
            }
        )
    pd.DataFrame(reverse_rows).to_csv(research_dir / "reverse_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "inst_id": "BTC-USDT-SWAP",
                "funding_data_complete": True,
                "row_count": 10,
                "trade_count": len(trades),
                "funding_pnl": 0.0,
            }
        ]
    ).to_csv(research_dir / "funding_summary.csv", index=False)
    for filename in [
        "event_group_summary.csv",
        "trade_group_summary.csv",
        "by_symbol.csv",
        "by_timeframe.csv",
        "by_split.csv",
        "concentration.csv",
    ]:
        pd.DataFrame([{"placeholder": "ok"}]).to_csv(research_dir / filename, index=False)


class PostmortemVsvcbV1Test(unittest.TestCase):
    def run_fixture(self, *, complete_inputs: bool = True) -> tuple[dict[str, object], Path]:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        research_dir = root / "research"
        output_dir = root / "postmortem"
        write_fixture(research_dir, complete_inputs=complete_inputs)
        summary = postmortem.run_postmortem(
            research_dir=research_dir,
            output_dir=output_dir,
            focus_group="D",
            reverse_group="E",
            primary_timeframe="15m",
            primary_symbols=SYMBOLS,
        )
        return summary, output_dir

    def test_missing_files_warning_does_not_crash(self) -> None:
        summary, output_dir = self.run_fixture(complete_inputs=False)

        self.assertTrue((output_dir / "vsvcb_v1_postmortem_summary.json").exists())
        self.assertTrue(any("missing input file" in item for item in summary["warnings"]))

    def test_d_vs_abc_less_loss_but_negative_is_failure(self) -> None:
        summary, output_dir = self.run_fixture()
        ablation = pd.read_csv(output_dir / "ablation_postmortem.csv")
        oos_rows = ablation[(ablation["timeframe"] == "15m") & (ablation["split"] == "oos")]

        self.assertTrue(summary["vsvcb_v1_failed"])
        self.assertTrue(summary["d_group"]["ablation_pass_but_no_edge"])
        self.assertTrue((oos_rows["d_no_cost_pnl"] < 0).all())
        self.assertTrue(oos_rows["d_better_than_baseline"].astype(bool).all())

    def test_reverse_e_better_marks_reverse_test_failure(self) -> None:
        summary, output_dir = self.run_fixture()
        reverse = pd.read_csv(output_dir / "reverse_directionality_postmortem.csv")
        oos = reverse[(reverse["timeframe"] == "15m") & (reverse["split"] == "oos")].iloc[0]

        self.assertTrue(summary["reverse_test_failure"])
        self.assertTrue(bool(oos["reverse_test_failure"]))
        self.assertGreater(float(oos["e_no_cost_pnl"]), float(oos["d_no_cost_pnl"]))

    def test_possible_false_breakout_research_hypothesis_true(self) -> None:
        summary, _output_dir = self.run_fixture()

        self.assertTrue(summary["possible_false_breakout_research_hypothesis"])

    def test_required_diagnostic_outputs_exist(self) -> None:
        _summary, output_dir = self.run_fixture()

        for filename in [
            "horizon_path_postmortem.csv",
            "feature_bin_postmortem.csv",
            "conflict_filter_impact.csv",
        ]:
            self.assertTrue((output_dir / filename).exists(), filename)

    def test_final_gates_all_false(self) -> None:
        summary, _output_dir = self.run_fixture()

        self.assertFalse(summary["continue_to_phase2"])
        self.assertFalse(summary["parameter_plateau_allowed"])
        self.assertFalse(summary["randomization_allowed"])
        self.assertFalse(summary["strategy_development_allowed"])
        self.assertFalse(summary["demo_live_allowed"])

    def test_makefile_target_exists(self) -> None:
        text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("postmortem-vsvcb-v1:", text)
        self.assertIn("scripts/postmortem_vsvcb_v1.py", text)


if __name__ == "__main__":
    unittest.main()
