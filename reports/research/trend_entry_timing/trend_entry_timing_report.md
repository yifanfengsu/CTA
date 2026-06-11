# Trend Entry Timing Research v1

## Scope
- This is research-only. Trend segment labels are ex-post labels and are not entry features.
- Candidate events use only closed-bar data available at event time.
- No Strategy V3, demo, or live permission is granted.

## Required Answers
1. Did legacy strategies enter too late? true (late_entry_share=0.736775).
2. Worst legacy policy / symbol / timeframe: policy={'policy_or_group': 'v3_4h_donchian_55_with_risk_filters', 'late_entry_share': 0.8630136986, 'avg_entry_lag_pct': 0.6101754014, 'trade_count': 73}, symbol={'symbol': 'BTCUSDT_SWAP_OKX.GLOBAL', 'late_entry_share': 0.7879581152, 'avg_entry_lag_pct': 0.5054777096, 'trade_count': 764}.
3. Do candidate early-entry events enter trends earlier? best_early_entry_family=relative_strength_leader (early_entry_rate=0.262568).
4. Highest recall family: relative_strength_leader (trend_segment_recall=0.469956).
5. Highest early_entry_rate family: relative_strength_leader.
6. Any family passed train/validation/oos, cost, and funding gates? false.
7. Reverse test weaker than forward? See rejected_candidate_entry_families.csv; required for gate.
8. Random time control weaker than forward? See rejected_candidate_entry_families.csv; required for gate.
9. If no family passes, current candidate features are insufficient; entry timing may still require a new hypothesis. passed=false.
10. Entry Timing Phase 2 allowed? false.
11. Formal strategy modification allowed? false.
12. Demo/live allowed? false.

## Candidate Families
| family | selected_hold_label | event_count | trend_segment_recall | early_entry_rate | direction_match_rate | train_ext_no_cost_pnl | validation_ext_no_cost_pnl | oos_ext_no_cost_pnl |
|---|---|---|---|---|---|---|---|---|
| pre_breakout_momentum_acceleration | fixed_hold_4h | 3779.000000 | 0.163188 | 0.158243 | 0.950781 | -25.667127 | 453.130801 | -708.611953 |
| breakout_retest_reclaim | fixed_hold_8h | 1307.000000 | 0.087919 | 0.171385 | 0.956389 | 93.180969 | -817.328925 | 1541.527756 |
| cross_symbol_breadth_acceleration | fixed_hold_4h | 7207.000000 | 0.384567 | 0.224504 | 0.889552 | 235.192977 | 3247.320238 | 1350.142764 |
| funding_neutral_momentum | fixed_hold_4h | 8054.000000 | 0.393422 | 0.252173 | 0.902036 | -1278.239517 | 402.149909 | 515.982326 |
| relative_strength_leader | fixed_hold_4h | 10165.000000 | 0.469956 | 0.262568 | 0.874176 | -616.811522 | 565.149216 | 75.842558 |

## Gate Rejections
| family | stable_like | selected_hold_label | rejected_reasons |
|---|---|---|---|
| pre_breakout_momentum_acceleration | false | fixed_hold_4h | train_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;trend_segment_recall_lt_0.20;early_entry_rate_lt_0.40 |
| breakout_retest_reclaim | false | fixed_hold_8h | validation_ext:no_cost_pnl_not_positive;trend_segment_recall_lt_0.20;early_entry_rate_lt_0.40 |
| cross_symbol_breadth_acceleration | false | fixed_hold_4h | oos_ext:cost_aware_pnl_negative;early_entry_rate_lt_0.40;largest_symbol_pnl_share_gt_0.7 |
| funding_neutral_momentum | false | fixed_hold_4h | train_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;early_entry_rate_lt_0.40;reverse_test_not_clearly_weaker;largest_symbol_pnl_share_gt_0.7 |
| relative_strength_leader | false | fixed_hold_4h | train_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;early_entry_rate_lt_0.40;reverse_test_not_clearly_weaker;random_time_control_not_clearly_weaker;largest_symbol_pnl_share_gt_0.7 |

## Final Gates
- can_enter_entry_timing_phase2=false
- strategy_development_allowed=false
- demo_live_allowed=false
