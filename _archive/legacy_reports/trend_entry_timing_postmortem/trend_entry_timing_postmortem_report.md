# Trend Entry Timing Postmortem & Breadth Candidate Audit

## Scope
- This is a postmortem, not strategy development.
- No candidate is marked tradable. Strategy modification, demo, and live remain disabled.
- Lower-cost diagnostics are sensitivity checks only; they are not a pass.

## Required Answers
1. Trend Entry Timing Research failed? true. can_enter_entry_timing_phase2 is fixed to false because no candidate passed the stable-like gates.
2. Why can_enter_entry_timing_phase2=false? Gate failures remain across no-cost stability, cost/funding, early-entry quality, concentration, or controls; see candidate_gate_postmortem.csv.
3. Is cross_symbol_breadth_acceleration a research asset? false.
4. Why can it be no-cost positive across splits but stable_like=false? OOS cost-aware=-13.8572, early_entry_rate=0.2245, largest_symbol_pnl_share=1.0000.
5. Is OOS cost-aware negative only marginal? marginal=true, structural_cost_drag=true, avg_trade_gross=0.9898, avg_trade_cost=1.0000.
6. Does funding-adjusted turn positive because of funding? funding_dependent=true, signed_funding_pnl=25.8043, conservative_adjusted=-21.9806.
7. Does it solve late entry? false. early_entry_rate=0.2245, median_entry_lag_pct=0.5000.
8. Is it better than random/reverse control? control_pass=true, reverse_or_random_stronger=false.
9. Is there concentration risk? true; largest_symbol_pnl_share=1.0000, remove_top_1_pnl=-1680.4461.
10. Phase 2 allowed? false.
11. Formal strategy modification allowed? false.
12. Demo/live allowed? false.
13. Recommended next step: pause_or_new_hypothesis.

## Gate Failure Summary
| family | primary_failure_category | failed_gate_count | train_no_cost | validation_no_cost | oos_no_cost | oos_cost | oos_funding | trend_recall | early_entry_rate | largest_symbol_pnl_share |
|---|---|---|---|---|---|---|---|---|---|---|
| pre_breakout_momentum_acceleration | no_cost_split_failure | 6.0000 | -25.6671 | 453.1308 | -708.6120 | -1354.6120 | -1341.9747 | 0.1632 | 0.1582 | 0.2947 |
| breakout_retest_reclaim | no_cost_split_failure | 3.0000 | 93.1810 | -817.3289 | 1541.5278 | 1307.5278 | 1300.9630 | 0.0879 | 0.1714 | 0.5831 |
| cross_symbol_breadth_acceleration | cost_or_execution_failure | 3.0000 | 235.1930 | 3247.3202 | 1350.1428 | -13.8572 | 11.9471 | 0.3846 | 0.2245 | 1.0000 |
| funding_neutral_momentum | no_cost_split_failure | 6.0000 | -1278.2395 | 402.1499 | 515.9823 | -1151.0177 | -1131.8713 | 0.3934 | 0.2522 | 1.0000 |
| relative_strength_leader | no_cost_split_failure | 7.0000 | -616.8115 | 565.1492 | 75.8426 | -1956.1574 | -1925.7469 | 0.4700 | 0.2626 | 1.0000 |

## Focus Family Deep Dive
| family | no_cost_all_splits_positive | oos_cost_aware_pnl | oos_funding_adjusted_pnl | funding_dependent | symbol_dependency | timeframe_dependency | top_trade_dependency | early_entry_quality | control_pass |
|---|---|---|---|---|---|---|---|---|---|
| cross_symbol_breadth_acceleration | true | -13.8572 | 11.9471 | true | true | true | true | insufficient_early_entry | true |

## Final Gates
- focus_family_research_asset=false
- can_enter_entry_timing_phase2=false
- strategy_development_allowed=false
- demo_live_allowed=false
- recommended_next_step=pause_or_new_hypothesis
