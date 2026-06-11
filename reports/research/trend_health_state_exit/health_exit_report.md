# Trend Health State Exit Research

## Scope
- This is offline counterfactual exit research. Legacy entries remain unchanged.
- Trend segment labels are used only for ex-post evaluation and oracle upper-bound diagnostics.
- No formal strategy modification, demo, or live permission is granted.

## Required Answers
1. What is the hypothesis? Hold only while trend health remains confirmed by efficiency, energy, drawdown, and time limits.
2. Did it materially improve captured_fraction? best_variant=health_drawdown_only, best_avg_capture=0.525435, original_avg_capture=0.470327, raw_improved=true, gate_material=false.
3. Did it materially reduce early_exit_share? best_early_exit_share=0.477094, original_early_exit_share=0.570742, raw_reduced=true, gate_material=false.
4. Is it better than original_exit? raw=true, gate_material=false.
5. Is it better than previously tested exits? See health_exit_vs_original.csv; previous non-oracle best is included when available.
6. Did any non-oracle variant pass train/validation/oos, cost, funding, and concentration gates? false.
7. Which health dimension worked best? Compare health_no_energy, health_drawdown_only, and health_energy_confirmed in the summary table.
8. Did volume energy help? If health_no_energy or health_energy_confirmed beats health_ema20_core, strict energy likely caused premature exits.
9. Is drawdown alone enough? See health_drawdown_only capture, early_exit_share, and gate rejection reasons.
10. If health exit failed, should research turn to entry timing? true.
11. Formal strategy modification allowed? false.
12. Demo/live allowed? false.

## Exit Variant Summary
| exit_variant | oracle | trade_count | train_ext_no_cost_pnl | validation_ext_no_cost_pnl | oos_ext_no_cost_pnl | oos_ext_cost_aware_pnl | oos_ext_funding_adjusted_pnl | avg_captured_fraction | early_exit_share | stable_like |
|---|---|---|---|---|---|---|---|---|---|---|
| health_drawdown_only | false | 3951.000000 | 5.044610 | 6.078650 | -32.268070 | -36.178085 | -37.273564 | 0.525435 | 0.477094 | false |
| health_ema20_core | false | 3951.000000 | 6.958910 | 4.573660 | -23.059320 | -26.969335 | -27.879289 | 0.284210 | 0.633004 | false |
| health_ema50_core | false | 3951.000000 | 10.679910 | 19.203150 | -20.771140 | -24.681155 | -25.884658 | 0.412423 | 0.578335 | false |
| health_energy_confirmed | false | 3951.000000 | 8.085430 | 3.686340 | -23.282420 | -27.192435 | -28.065978 | 0.290474 | 0.630473 | false |
| health_no_energy | false | 3951.000000 | -1.491610 | 3.222690 | -24.054320 | -27.964335 | -28.581530 | 0.191485 | 0.664136 | false |
| original_exit | false | 3951.000000 | 7.973160 | -15.665470 | -32.129780 | -36.039795 | -37.361312 | 0.470327 | 0.570742 | false |
| oracle_hold_to_trend_end | true | 3951.000000 | -15.203360 | -27.956170 | -129.456620 | -133.366635 | -134.921409 | 0.738226 | 0.276133 | false |

## Gate Rejections
| exit_variant | oracle | stable_like | rejected_reasons | avg_captured_fraction | early_exit_share |
|---|---|---|---|---|---|
| health_drawdown_only | false | false | oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;largest_symbol_pnl_share_gt_0.7;avg_captured_fraction_lt_original_plus_0.15;early_exit_share_not_reduced_by_0.20 | 0.525435 | 0.477094 |
| health_ema20_core | false | false | oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;avg_captured_fraction_lt_original_plus_0.15;early_exit_share_not_reduced_by_0.20 | 0.284210 | 0.633004 |
| health_ema50_core | false | false | oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;largest_symbol_pnl_share_gt_0.7;avg_captured_fraction_lt_original_plus_0.15;early_exit_share_not_reduced_by_0.20 | 0.412423 | 0.578335 |
| health_energy_confirmed | false | false | oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;avg_captured_fraction_lt_original_plus_0.15;early_exit_share_not_reduced_by_0.20 | 0.290474 | 0.630473 |
| health_no_energy | false | false | train_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;avg_captured_fraction_lt_original_plus_0.15;early_exit_share_not_reduced_by_0.20 | 0.191485 | 0.664136 |
| original_exit | false | false | original_exit_reference_excluded_from_health_gate;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;avg_captured_fraction_lt_original_plus_0.15;early_exit_share_not_reduced_by_0.20 | 0.470327 | 0.570742 |
| oracle_hold_to_trend_end | true | false | oracle_variant_excluded_from_stable_gate;train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative | 0.738226 | 0.276133 |

## Final Gates
- can_enter_health_exit_phase2=false
- strategy_development_allowed=false
- demo_live_allowed=false
- recommended_next_step=entry_timing_research_or_pause
