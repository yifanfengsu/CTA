# Trend Capture & Exit Convexity Research

## Scope
- This is counterfactual exit research only, not strategy development.
- Legacy entries are unchanged. Trend segment labels are ex-post diagnostics only.
- oracle_hold_to_trend_end is an upper-bound diagnostic and is not tradable.
- strategy_development_allowed=false
- demo_live_allowed=false

## Required Answers
1. Is there trend opportunity in the market? true.
2. Did legacy strategies enter trend segments? entered_share=0.993467.
3. Were legacy strategies late, early out, or both? main_failure_mode=entered_middle_or_late, late_entry_share=0.634330, early_exit_share=0.961308.
4. Is avg_captured_fraction materially low? avg_captured_fraction=0.068769; yes.
5. Is early_exit_share materially high? early_exit_share=0.961308; yes.
6. Can exit-only changes make any legacy policy pass the stable-like gate? false.
7. Which exit variant improved the most? atr_chandelier_5x (capture_improvement=0.397798).
8. How high is oracle_hold_to_trend_end upper bound? no_cost=1718325.409421, funding_adjusted=1535040.987874, avg_capture=0.514850.
9. If oracle is good but tradable exits are not, it means exits may help in theory but the tradable path control still cannot harvest the ex-post trend end without better timing or risk logic.
10. If all tradable exits fail, does it point mainly to entry timing? true.
11. Is Exit Convexity Phase 2 allowed? false.
12. Is formal strategy modification allowed? false.
13. Is demo/live allowed? false.

## Exit Variant Summary
| exit_variant | oracle | trade_count | no_cost_pnl | cost_aware_pnl | funding_adjusted_pnl | avg_captured_fraction | early_exit_share | late_entry_share |
|---|---|---|---|---|---|---|---|---|
| atr_chandelier_3x | false | 78672.000000 | -732980.833925 | -882433.821550 | -934612.909799 | 0.406549 | 0.652824 | 0.634330 |
| atr_chandelier_5x | false | 78672.000000 | -1500289.970510 | -1649742.958135 | -1747046.645329 | 0.466567 | 0.574334 | 0.634330 |
| fixed_hold_longer_2x | false | 78672.000000 | -310405.179751 | -459858.167375 | -460351.533855 | 0.119611 | 0.932759 | 0.634330 |
| fixed_hold_longer_4x | false | 78672.000000 | -293365.917438 | -442818.905062 | -444286.060311 | 0.170005 | 0.902990 | 0.634330 |
| oracle_hold_to_trend_end | true | 78672.000000 | 1718325.409421 | 1568872.421796 | 1535040.987874 | 0.514850 | 0.493924 | 0.634330 |
| original_exit | false | 78672.000000 | -332658.075803 | -482111.063428 | -482182.439635 | 0.068769 | 0.961308 | 0.634330 |
| swing_trailing_exit | false | 78672.000000 | -527499.898337 | -676952.885962 | -690418.807305 | 0.212647 | 0.864844 | 0.634330 |
| time_stop_if_no_progress | false | 78672.000000 | -526771.933248 | -676224.920872 | -676521.186264 | 0.085947 | 0.952474 | 0.634330 |

## Gate Rejections
| exit_variant | oracle | stable_like | rejected_reasons |
|---|---|---|---|
| atr_chandelier_3x | false | false | train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative |
| atr_chandelier_5x | false | false | train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;oos_ext:largest_symbol_pnl_share_gt_0.7 |
| fixed_hold_longer_2x | false | false | train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;oos_ext:largest_symbol_pnl_share_gt_0.7;early_exit_share_not_materially_lower_than_original |
| fixed_hold_longer_4x | false | false | train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;oos_ext:largest_symbol_pnl_share_gt_0.7 |
| oracle_hold_to_trend_end | true | false | oracle_variant_excluded_from_stable_gate;oos_ext:top_5pct_trade_pnl_contribution_gt_0.8 |
| original_exit | false | false | train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;oos_ext:largest_symbol_pnl_share_gt_0.7;avg_captured_fraction_not_materially_higher_than_original;early_exit_share_not_materially_lower_than_original |
| swing_trailing_exit | false | false | train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative |
| time_stop_if_no_progress | false | false | train_ext:no_cost_pnl_not_positive;validation_ext:no_cost_pnl_not_positive;oos_ext:no_cost_pnl_not_positive;oos_ext:cost_aware_pnl_negative;oos_ext:funding_adjusted_pnl_negative;early_exit_share_not_materially_lower_than_original |
